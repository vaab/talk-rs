//! Bounded background loading of waterfall spectrograms for the
//! recordings browser.
//!
//! Every row without a transcript shows a waterfall, computed by
//! [`crate::record::audio::load_waterfall`] (cache hit, or decode +
//! FFT).  A library with hundreds of such rows must not start hundreds
//! of threads at once: [`WaterfallLoader`] queues the requests in row
//! order (newest first, as the browser lists them) and runs at most
//! [`WATERFALL_WORKERS`] of them at a time.  Nothing is dropped: a
//! queued request waits its turn, and its result is applied to the row
//! when the GTK thread next drains the loader.
//!
//! A request is tied to the row's interest in it ([`Interest`]): when
//! the row widget goes away before its waterfall is ready, dropping
//! the handle cancels the queued job (or discards its result), so a
//! refreshed or deleted row never costs a decode nobody will see.
//!
//! The loader itself has no GTK dependency (it is unit-tested with a
//! fake work function); [`request`] is the browser-side entry point
//! that uses the process-wide loader and polls it from the GTK main
//! loop while anything is outstanding.

use crate::error::TalkError;
use crate::perf_counters::{self, Counter, Gauge};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};

/// Waterfall column data: (columns, peak).
pub(crate) type WfColumns = (Vec<Vec<f32>>, f32);

/// Waterfall computations running at the same time.  Four keeps a
/// cold library from starving the GTK thread while still filling the
/// visible rows quickly; the rest queue in row order.
pub(crate) const WATERFALL_WORKERS: usize = 4;

/// Result of one waterfall computation.
pub(crate) type WfResult = Result<WfColumns, TalkError>;

/// The computation itself: `load_waterfall` in production, a scripted
/// function in tests.
type Work = Arc<dyn Fn(&Path) -> WfResult + Send + Sync>;

/// Applied on the requesting thread when a result arrives.
type Apply = Box<dyn FnOnce(WfResult)>;

struct Job {
    id: u64,
    path: PathBuf,
    cancelled: Arc<AtomicBool>,
}

/// State shared with the worker threads.
struct Shared {
    queue: Mutex<VecDeque<Job>>,
    wake: Condvar,
    shutdown: AtomicBool,
    work: Work,
    results: Mutex<mpsc::Sender<(u64, WfResult)>>,
}

struct Inner {
    shared: Arc<Shared>,
    results: mpsc::Receiver<(u64, WfResult)>,
    pending: RefCell<HashMap<u64, Apply>>,
    next_id: Cell<u64>,
    workers: usize,
    spawned: Cell<bool>,
    workers_started: Cell<usize>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        self.shared.wake.notify_all();
    }
}

/// A queue of waterfall requests served by a fixed number of worker
/// threads, drained by the thread that owns it.
pub(crate) struct WaterfallLoader {
    inner: Rc<Inner>,
}

/// A row's interest in a requested waterfall.  Dropping it cancels
/// the request: a job still queued never starts, a running one has its
/// result discarded, and the apply callback is never called.
pub(crate) struct Interest {
    id: u64,
    cancelled: Arc<AtomicBool>,
    loader: Weak<Inner>,
}

impl Drop for Interest {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Release);
        if let Some(inner) = self.loader.upgrade() {
            inner.pending.borrow_mut().remove(&self.id);
        }
    }
}

impl WaterfallLoader {
    /// A loader running at most `workers` computations at a time.
    /// Worker threads start with the first request.
    pub(crate) fn new(workers: usize, work: Work) -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            inner: Rc::new(Inner {
                shared: Arc::new(Shared {
                    queue: Mutex::new(VecDeque::new()),
                    wake: Condvar::new(),
                    shutdown: AtomicBool::new(false),
                    work,
                    results: Mutex::new(tx),
                }),
                results: rx,
                pending: RefCell::new(HashMap::new()),
                next_id: Cell::new(1),
                workers: workers.max(1),
                spawned: Cell::new(false),
                workers_started: Cell::new(0),
            }),
        }
    }

    /// Queue the waterfall of `path`.  `apply` runs on this thread,
    /// from [`deliver_ready`](Self::deliver_ready), once the result is
    /// in — unless the returned [`Interest`] was dropped first.
    pub(crate) fn request(&self, path: &Path, apply: Apply) -> Interest {
        let inner = &self.inner;
        inner.spawn_workers();
        let id = inner.next_id.get();
        inner.next_id.set(id + 1);
        let cancelled = Arc::new(AtomicBool::new(false));
        inner.pending.borrow_mut().insert(id, apply);
        perf_counters::incr(Counter::WaterfallJobsRequested);
        if inner.workers_started.get() == 0 {
            if let Some(apply) = inner.pending.borrow_mut().remove(&id) {
                apply(Err(TalkError::Audio("waterfall worker unavailable".into())));
            }
            return Interest {
                id,
                cancelled,
                loader: Rc::downgrade(inner),
            };
        }
        if let Ok(mut queue) = inner.shared.queue.lock() {
            queue.push_back(Job {
                id,
                path: path.to_path_buf(),
                cancelled: Arc::clone(&cancelled),
            });
        }
        inner.shared.wake.notify_one();
        Interest {
            id,
            cancelled,
            loader: Rc::downgrade(inner),
        }
    }

    /// Apply a short slice of ready results; the timer drains the rest later.
    /// Results whose interest was dropped are discarded.
    pub(crate) fn deliver_ready(&self) -> usize {
        let started = std::time::Instant::now();
        let mut applied = 0;
        for _ in 0..4 {
            if started.elapsed() >= std::time::Duration::from_millis(5) {
                break;
            }
            let Ok((id, result)) = self.inner.results.try_recv() else {
                break;
            };
            let apply = self.inner.pending.borrow_mut().remove(&id);
            if let Some(apply) = apply {
                apply(result);
                applied += 1;
            }
        }
        applied
    }

    /// Requests whose result has not been applied yet.
    pub(crate) fn pending_count(&self) -> usize {
        self.inner.pending.borrow().len()
    }
}

impl Inner {
    fn spawn_workers(&self) {
        if self.spawned.replace(true) {
            return;
        }
        for n in 0..self.workers {
            let shared = Arc::clone(&self.shared);
            let spawned = std::thread::Builder::new()
                .name(format!("waterfall-{n}"))
                .spawn(move || worker(&shared));
            match spawned {
                Ok(_) => self.workers_started.set(self.workers_started.get() + 1),
                Err(e) => log::warn!("waterfall: cannot start worker {n}: {e}"),
            }
        }
    }
}

fn worker(shared: &Shared) {
    loop {
        let job = {
            let Ok(mut queue) = shared.queue.lock() else {
                return;
            };
            loop {
                if shared.shutdown.load(Ordering::Acquire) {
                    return;
                }
                if let Some(job) = queue.pop_front() {
                    break job;
                }
                queue = match shared.wake.wait(queue) {
                    Ok(q) => q,
                    Err(_) => return,
                };
            }
        };
        if job.cancelled.load(Ordering::Acquire) {
            continue;
        }
        perf_counters::gauge_inc(Gauge::WaterfallWorkersInflight);
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (shared.work)(&job.path)))
                .unwrap_or_else(|_| {
                    Err(TalkError::Audio(format!(
                        "waterfall worker panicked: {}",
                        job.path.display()
                    )))
                });
        perf_counters::gauge_dec(Gauge::WaterfallWorkersInflight);
        if job.cancelled.load(Ordering::Acquire) {
            continue;
        }
        if let Ok(tx) = shared.results.lock() {
            // A closed channel means the loader is gone: nothing to do.
            let _ = tx.send((job.id, result));
        }
    }
}

thread_local! {
    /// The recordings browser's loader (one per process: GTK runs on
    /// one thread).
    static LOADER: WaterfallLoader = WaterfallLoader::new(
        WATERFALL_WORKERS,
        Arc::new(crate::record::audio::load_waterfall),
    );
    /// Whether the 50 ms drain timer is registered.
    static POLLING: Cell<bool> = const { Cell::new(false) };
}

/// Queue a waterfall for the browser and make sure results get
/// applied: one 50 ms GTK timer drains the loader while requests are
/// outstanding and removes itself once none are.
pub(crate) fn request(path: &Path, apply: Apply) -> Interest {
    use gtk4::glib;
    let interest = LOADER.with(|loader| loader.request(path, apply));
    if !POLLING.replace(true) {
        glib::timeout_add_local(std::time::Duration::from_millis(50), || {
            let outstanding = LOADER.with(|loader| {
                loader.deliver_ready();
                loader.pending_count()
            });
            if outstanding > 0 {
                glib::ControlFlow::Continue
            } else {
                POLLING.set(false);
                glib::ControlFlow::Break
            }
        });
    }
    interest
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::time::{Duration, Instant};

    fn columns(tag: f32) -> WfColumns {
        (vec![vec![tag; 2]], tag)
    }

    #[test]
    fn completed_waveforms_yield_to_gtk_in_bounded_slices() {
        let finished = Arc::new(AtomicUsize::new(0));
        let work: Work = {
            let finished = Arc::clone(&finished);
            Arc::new(move |_| {
                finished.fetch_add(1, Ordering::SeqCst);
                Ok(columns(0.0))
            })
        };
        let loader = WaterfallLoader::new(1, work);
        let _interests: Vec<_> = (0..12)
            .map(|i| loader.request(Path::new(&format!("{i}.ogg")), Box::new(|_| {})))
            .collect();
        let deadline = Instant::now() + Duration::from_secs(5);
        while finished.load(Ordering::SeqCst) < 12 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(finished.load(Ordering::SeqCst), 12);
        let first = loader.deliver_ready();
        assert!(
            (1..12).contains(&first),
            "one GTK turn must yield before draining all results"
        );
        assert_eq!(first + drain(&loader, 12 - first), 12);
    }

    /// Drain the loader until `n` results were applied or 5 s pass.
    fn drain(loader: &WaterfallLoader, n: usize) -> usize {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut applied = 0;
        while applied < n && Instant::now() < deadline {
            applied += loader.deliver_ready();
            std::thread::sleep(Duration::from_millis(2));
        }
        applied
    }

    #[test]
    fn at_most_the_configured_number_of_jobs_run_at_once() {
        let inflight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let work: Work = {
            let (inflight, peak) = (Arc::clone(&inflight), Arc::clone(&peak));
            Arc::new(move |_: &Path| {
                let now = inflight.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                std::thread::sleep(Duration::from_millis(20));
                inflight.fetch_sub(1, Ordering::SeqCst);
                Ok(columns(1.0))
            })
        };
        let loader = WaterfallLoader::new(2, work);
        let applied = Rc::new(Cell::new(0));
        let mut interests = Vec::new();
        for i in 0..8 {
            let applied = Rc::clone(&applied);
            interests.push(loader.request(
                Path::new(&format!("row-{i}.ogg")),
                Box::new(move |r| {
                    assert!(r.is_ok());
                    applied.set(applied.get() + 1);
                }),
            ));
        }
        assert_eq!(drain(&loader, 8), 8, "every request is applied");
        assert_eq!(applied.get(), 8);
        assert_eq!(
            peak.load(Ordering::SeqCst),
            2,
            "never more than 2 in flight"
        );
        assert_eq!(loader.pending_count(), 0);
    }

    #[test]
    fn jobs_start_in_request_order() {
        let started = Arc::new(Mutex::new(Vec::new()));
        let work: Work = {
            let started = Arc::clone(&started);
            Arc::new(move |p: &Path| {
                if let Ok(mut s) = started.lock() {
                    s.push(p.to_path_buf());
                }
                Ok(columns(0.0))
            })
        };
        let loader = WaterfallLoader::new(1, work);
        let paths: Vec<PathBuf> = (0..5).map(|i| PathBuf::from(format!("{i}.ogg"))).collect();
        let _interests: Vec<Interest> = paths
            .iter()
            .map(|p| loader.request(p, Box::new(|_| {})))
            .collect();
        assert_eq!(drain(&loader, 5), 5);
        assert_eq!(started.lock().map(|s| s.clone()).unwrap_or_default(), paths);
    }

    #[test]
    fn dropping_the_interest_cancels_a_queued_job() {
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let release_rx = Arc::new(Mutex::new(release_rx));
        let ran = Arc::new(Mutex::new(Vec::new()));
        let work: Work = {
            let (ran, release_rx) = (Arc::clone(&ran), Arc::clone(&release_rx));
            Arc::new(move |p: &Path| {
                if p == Path::new("blocker.ogg") {
                    // Hold the single worker until the test releases it.
                    let _ = release_rx.lock().map(|rx| rx.recv());
                }
                if let Ok(mut r) = ran.lock() {
                    r.push(p.to_path_buf());
                }
                Ok(columns(0.0))
            })
        };
        let loader = WaterfallLoader::new(1, work);
        let applied = Rc::new(RefCell::new(Vec::new()));
        let apply = |tag: &'static str| -> Apply {
            let applied = Rc::clone(&applied);
            Box::new(move |_| applied.borrow_mut().push(tag))
        };
        let _blocker = loader.request(Path::new("blocker.ogg"), apply("blocker"));
        let dropped = loader.request(Path::new("dropped.ogg"), apply("dropped"));
        let _kept = loader.request(Path::new("kept.ogg"), apply("kept"));
        assert_eq!(loader.pending_count(), 3);
        drop(dropped);
        assert_eq!(loader.pending_count(), 2, "cancelling forgets the request");
        release_tx.send(()).expect("release the worker");
        assert_eq!(drain(&loader, 2), 2);
        assert_eq!(
            ran.lock().map(|r| r.clone()).unwrap_or_default(),
            vec![PathBuf::from("blocker.ogg"), PathBuf::from("kept.ogg")],
            "the cancelled job never ran"
        );
        assert_eq!(*applied.borrow(), vec!["blocker", "kept"]);
    }

    #[test]
    fn a_result_whose_interest_was_dropped_is_discarded() {
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let release_rx = Arc::new(Mutex::new(release_rx));
        let work: Work = {
            let release_rx = Arc::clone(&release_rx);
            Arc::new(move |_: &Path| {
                let _ = release_rx.lock().map(|rx| rx.recv());
                Ok(columns(0.0))
            })
        };
        let loader = WaterfallLoader::new(1, work);
        let applied = Rc::new(Cell::new(false));
        let interest = loader.request(Path::new("running.ogg"), {
            let applied = Rc::clone(&applied);
            Box::new(move |_| applied.set(true))
        });
        // Let the worker pick the job up, then lose interest mid-run.
        std::thread::sleep(Duration::from_millis(20));
        drop(interest);
        release_tx.send(()).expect("release the worker");
        // Give the worker ample time to finish and post its result.
        let deadline = Instant::now() + Duration::from_millis(300);
        let mut applied_count = 0;
        while Instant::now() < deadline {
            applied_count += loader.deliver_ready();
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(applied_count, 0, "nothing to apply");
        assert!(!applied.get());
        assert_eq!(loader.pending_count(), 0);
    }

    #[test]
    fn a_failed_computation_is_delivered_as_an_error() {
        let work: Work = Arc::new(|p: &Path| Err(TalkError::Audio(format!("bad {}", p.display()))));
        let loader = WaterfallLoader::new(1, work);
        let seen = Rc::new(RefCell::new(None));
        let _interest = loader.request(Path::new("broken.ogg"), {
            let seen = Rc::clone(&seen);
            Box::new(move |r| *seen.borrow_mut() = Some(r.map(|_| ()).map_err(|e| e.to_string())))
        });
        assert_eq!(drain(&loader, 1), 1);
        let seen = seen.borrow();
        let err = seen.as_ref().and_then(|r| r.as_ref().err());
        assert!(
            err.is_some_and(|e| e.contains("bad broken.ogg")),
            "error delivered: {seen:?}"
        );
    }

    #[test]
    fn a_panicking_job_does_not_strand_the_next_waveform() {
        let work: Work = Arc::new(|path| {
            if path == Path::new("bad.ogg") {
                panic!("decoder failure");
            }
            Ok(columns(1.0))
        });
        let loader = WaterfallLoader::new(1, work);
        let observed = Rc::new(RefCell::new(Vec::new()));
        let interests: Vec<_> = ["bad.ogg", "good.ogg"]
            .iter()
            .map(|path| {
                let seen = Rc::clone(&observed);
                loader.request(
                    Path::new(path),
                    Box::new(move |result| {
                        seen.borrow_mut().push(result.is_ok());
                    }),
                )
            })
            .collect();
        assert_eq!(drain(&loader, 2), 2);
        assert_eq!(*observed.borrow(), [false, true]);
        drop(interests);
    }
}
