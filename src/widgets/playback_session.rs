//! Playback shared by every row of a recordings window.
//!
//! One window has one audio player, at most one row that owns it, and
//! one progress tick.  [`PlaybackCore`] is the GTK-free state machine
//! behind that: which row is active, the decode in flight for it, a
//! pause or seek asked for before the decode finished, and the
//! interpolated cursor position.  [`PlaybackSession`] wraps it for the
//! GTK thread, registering the 16 ms tick only while a row is active
//! and dropping it as soon as playback finishes or is released, so an
//! idle window with hundreds of player bars runs no timer at all.
//!
//! Rows take part through [`PlaybackRow`]: the session tells the
//! active row its progress and tells a row it no longer owns playback.
//! A row that is removed from the list while inactive is simply
//! dropped — nothing here holds it.
//!
//! Play never decodes on the calling thread: the transport starts the
//! decode on a worker ([`Transport::start_load`]) and the tick loads
//! the samples when they arrive, applying any pause or seek requested
//! meanwhile.  A new Play while a decode is still running supersedes
//! it; the stale result is discarded.

use crate::record::player::{LoadPoll, PendingLoad, WavPlayer};
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;

/// What the session needs from the audio player.
pub(crate) trait Transport {
    /// Start decoding `path` on a worker thread.
    fn start_load(&self, path: &Path) -> PendingLoad;
    /// Load decoded samples and start playing from the beginning.
    fn load_at(&self, samples: Vec<f32>, fraction: f64, paused: bool);
    /// Stop and drop the loaded samples.
    fn stop(&self);
    fn pause(&self);
    fn resume(&self);
    fn seek(&self, fraction: f64);
    /// Raw position as a fraction of the loaded samples.
    fn progress(&self) -> f64;
    fn duration_secs(&self) -> f64;
    /// `true` when every sample was consumed (or nothing is loaded).
    fn is_finished(&self) -> bool;
}

impl Transport for WavPlayer {
    fn start_load(&self, path: &Path) -> PendingLoad {
        self.load_in_background(path)
    }
    fn load_at(&self, samples: Vec<f32>, fraction: f64, paused: bool) {
        WavPlayer::load_at(self, samples, fraction, paused);
    }
    fn stop(&self) {
        WavPlayer::stop(self);
    }
    fn pause(&self) {
        WavPlayer::pause(self);
    }
    fn resume(&self) {
        WavPlayer::resume(self);
    }
    fn seek(&self, fraction: f64) {
        WavPlayer::seek(self, fraction);
    }
    fn progress(&self) -> f64 {
        WavPlayer::progress(self)
    }
    fn duration_secs(&self) -> f64 {
        WavPlayer::duration_secs(self)
    }
    fn is_finished(&self) -> bool {
        WavPlayer::is_finished(self)
    }
}

/// The row-side view of playback: what the session updates on the row
/// that owns the player.
pub(crate) trait PlaybackRow {
    /// The row no longer owns playback.  `finished` tells a natural end
    /// (cursor back to the start) from a release by another row or a
    /// stop (cursor kept where it was, so Play resumes from there).
    fn set_idle(&self, finished: bool);
    /// Playback position, 0.0–1.0, while playing.
    fn set_progress(&self, fraction: f64);
}

/// Shared handle to a row.
pub(crate) type RowRef = Rc<dyn PlaybackRow>;

fn same_row(a: &RowRef, b: &RowRef) -> bool {
    std::ptr::addr_eq(Rc::as_ptr(a), Rc::as_ptr(b))
}

/// What the caller of [`PlaybackCore::tick`] should do next.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Tick {
    /// A row is active: keep ticking.
    Active,
    /// Paused with no decode pending: retain the row without polling.
    Dormant,
    /// Nothing is active: the tick can stop.
    Idle,
}

/// The playback state machine (no GTK).
pub(crate) struct PlaybackCore<T: Transport> {
    transport: Option<T>,
    active: Option<RowRef>,
    active_path: Option<PathBuf>,
    load: Option<PendingLoad>,
    pending_seek: Option<f64>,
    paused: bool,
    /// Last raw progress seen and when: the cursor is extrapolated from
    /// it between the output callback's buffer-sized jumps.
    interp: (f64, Instant),
}

impl<T: Transport> Default for PlaybackCore<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Transport> PlaybackCore<T> {
    pub(crate) fn new() -> Self {
        Self {
            transport: None,
            active: None,
            active_path: None,
            load: None,
            pending_seek: None,
            paused: false,
            interp: (0.0, Instant::now()),
        }
    }

    /// Install the player once the output device is open.
    pub(crate) fn set_transport(&mut self, transport: T) {
        self.transport = Some(transport);
    }

    pub(crate) fn has_transport(&self) -> bool {
        self.transport.is_some()
    }

    /// `true` when `row` owns playback (loading, playing or paused).
    pub(crate) fn is_active(&self, row: &RowRef) -> bool {
        self.active.as_ref().is_some_and(|a| same_row(a, row))
    }

    pub(crate) fn is_paused(&self) -> bool {
        self.active.is_some() && self.paused
    }

    /// A row owns playback and is not paused (its decode may still be
    /// running).
    pub(crate) fn is_playing(&self) -> bool {
        self.active.is_some() && !self.paused
    }

    /// Give playback of `path` to `row`, starting from `from` (a
    /// fraction strictly inside the file) or the beginning.  Any
    /// previous owner is released at once.  Returns `false` when no
    /// player is available.
    pub(crate) fn play(&mut self, row: RowRef, path: &Path, from: Option<f64>) -> bool {
        let Some(transport) = &self.transport else {
            return false;
        };
        if let Some(previous) = self.active.take() {
            if !same_row(&previous, &row) {
                previous.set_idle(false);
            }
        }
        transport.stop();
        self.load = Some(transport.start_load(path));
        self.pending_seek = from.filter(|f| *f > 0.0 && *f < 1.0);
        self.paused = false;
        self.interp = (self.pending_seek.unwrap_or(0.0), Instant::now());
        self.active = Some(row);
        self.active_path = Some(path.to_path_buf());
        true
    }

    /// A same-file row replacement takes over the existing transport.
    pub(crate) fn adopt(&mut self, row: RowRef, path: &Path) -> bool {
        if self.active_path.as_deref() != Some(path) {
            return false;
        }
        if let Some(previous) = self.active.replace(row) {
            previous.set_idle(false);
        }
        true
    }

    pub(crate) fn release_if_active(&mut self, row: &RowRef) -> bool {
        if !self.is_active(row) {
            return false;
        }
        self.stop();
        true
    }

    pub(crate) fn pause(&mut self) {
        if self.active.is_none() {
            return;
        }
        self.paused = true;
        if let Some(t) = &self.transport {
            t.pause();
        }
    }

    pub(crate) fn resume(&mut self) {
        if self.active.is_none() {
            return;
        }
        self.paused = false;
        if let Some(t) = &self.transport {
            t.resume();
        }
    }

    /// Seek the active playback; while its decode is still running the
    /// position is applied once the samples arrive.
    pub(crate) fn seek(&mut self, fraction: f64) {
        if self.active.is_none() {
            return;
        }
        let fraction = fraction.clamp(0.0, 1.0);
        if self.load.is_some() {
            self.pending_seek = Some(fraction);
        } else if let Some(t) = &self.transport {
            t.seek(fraction);
        }
        self.interp = (fraction, Instant::now());
    }

    /// Release the active row and stop the player.
    pub(crate) fn stop(&mut self) {
        self.release(false);
        if let Some(t) = &self.transport {
            t.stop();
        }
    }

    fn release(&mut self, finished: bool) {
        if let Some(row) = self.active.take() {
            row.set_idle(finished);
        }
        self.load = None;
        self.active_path = None;
        self.pending_seek = None;
        self.paused = false;
    }

    /// One step of the progress tick, at `now`: finish a pending load,
    /// notice the end of playback, or push the cursor position.
    pub(crate) fn tick(&mut self, now: Instant) -> Tick {
        let Some(transport) = &self.transport else {
            return Tick::Idle;
        };
        let Some(row) = &self.active else {
            return Tick::Idle;
        };
        if let Some(load) = &self.load {
            match load.try_take() {
                LoadPoll::Pending => return Tick::Active,
                LoadPoll::Ready(samples) => {
                    self.load = None;
                    transport.load_at(
                        samples,
                        self.pending_seek.take().unwrap_or(0.0),
                        self.paused,
                    );
                    self.interp = (transport.progress(), now);
                }
                LoadPoll::Failed(e) => {
                    log::warn!("failed to play audio: {e}");
                    self.release(false);
                    return Tick::Idle;
                }
            }
        }
        if transport.is_finished() {
            transport.stop();
            self.release(true);
            return Tick::Idle;
        }
        let raw = transport.progress();
        if self.paused {
            // Keep the reference fresh so resuming does not jump.
            self.interp = (raw, now);
            return Tick::Dormant;
        }
        if (raw - self.interp.0).abs() > 1e-9 {
            self.interp = (raw, now);
        }
        let duration = transport.duration_secs();
        let fraction = if duration > 0.0 {
            let elapsed = now.saturating_duration_since(self.interp.1).as_secs_f64();
            (self.interp.0 + elapsed / duration).min(1.0)
        } else {
            raw
        };
        row.set_progress(fraction);
        Tick::Active
    }
}

/// The GTK-thread session: the core plus the one progress tick,
/// registered while a row is active.
pub(crate) struct PlaybackSession {
    core: RefCell<PlaybackCore<WavPlayer>>,
    tick: RefCell<Option<gtk4::glib::SourceId>>,
}

/// Cursor refresh period (~60 fps).
const TICK_MS: u64 = 16;

impl PlaybackSession {
    pub(crate) fn new() -> Rc<Self> {
        Rc::new(Self {
            core: RefCell::new(PlaybackCore::new()),
            tick: RefCell::new(None),
        })
    }

    /// Install the player once the output device is open.
    pub(crate) fn set_player(&self, player: WavPlayer) {
        self.core.borrow_mut().set_transport(player);
    }

    pub(crate) fn has_player(&self) -> bool {
        self.core.borrow().has_transport()
    }

    pub(crate) fn is_active(&self, row: &RowRef) -> bool {
        self.core.borrow().is_active(row)
    }

    pub(crate) fn is_paused(&self) -> bool {
        self.core.borrow().is_paused()
    }

    pub(crate) fn is_playing(&self) -> bool {
        self.core.borrow().is_playing()
    }

    pub(crate) fn adopt(&self, row: RowRef, path: &Path) -> bool {
        self.core.borrow_mut().adopt(row, path)
    }

    pub(crate) fn release_if_active(&self, row: &RowRef) {
        if self.core.borrow_mut().release_if_active(row) {
            self.remove_tick();
        }
    }

    /// See [`PlaybackCore::play`]; also starts the tick.
    pub(crate) fn play(self: &Rc<Self>, row: RowRef, path: &Path, from: Option<f64>) -> bool {
        let started = self.core.borrow_mut().play(row, path, from);
        if started {
            self.ensure_tick();
        }
        started
    }

    pub(crate) fn pause(&self) {
        self.core.borrow_mut().pause();
    }

    pub(crate) fn resume(self: &Rc<Self>) {
        self.core.borrow_mut().resume();
        if self.core.borrow().is_playing() {
            self.ensure_tick();
        }
    }

    pub(crate) fn seek(&self, fraction: f64) {
        self.core.borrow_mut().seek(fraction);
    }

    /// Release the active row and stop the player; the tick notices on
    /// its next callback and removes itself.
    pub(crate) fn stop(&self) {
        self.core.borrow_mut().stop();
        self.remove_tick();
    }

    fn remove_tick(&self) {
        if let Some(id) = self.tick.borrow_mut().take() {
            id.remove();
            crate::perf_counters::gauge_dec(crate::perf_counters::Gauge::PlayerTickSourcesLive);
        }
    }

    fn ensure_tick(self: &Rc<Self>) {
        use gtk4::glib;
        if self.tick.borrow().is_some() {
            return;
        }
        crate::perf_counters::gauge_inc(crate::perf_counters::Gauge::PlayerTickSourcesLive);
        let weak = Rc::downgrade(self);
        let id = glib::timeout_add_local(std::time::Duration::from_millis(TICK_MS), move || {
            crate::perf_counters::incr(crate::perf_counters::Counter::PlayerTickCallbacks);
            let outcome = weak.upgrade().map(|session| {
                let step = session.core.borrow_mut().tick(Instant::now());
                (session, step)
            });
            match outcome {
                Some((_, Tick::Active)) => glib::ControlFlow::Continue,
                Some((session, Tick::Idle | Tick::Dormant)) => {
                    *session.tick.borrow_mut() = None;
                    crate::perf_counters::gauge_dec(
                        crate::perf_counters::Gauge::PlayerTickSourcesLive,
                    );
                    glib::ControlFlow::Break
                }
                None => {
                    crate::perf_counters::gauge_dec(
                        crate::perf_counters::Gauge::PlayerTickSourcesLive,
                    );
                    glib::ControlFlow::Break
                }
            }
        });
        *self.tick.borrow_mut() = Some(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::TalkError;
    use std::sync::mpsc;
    use std::time::Duration;

    #[derive(Default)]
    struct FakeState {
        samples: usize,
        position: usize,
        paused: bool,
        loads: usize,
        stops: usize,
        /// Senders of decodes started, oldest first.
        decodes: Vec<mpsc::Sender<Result<Vec<f32>, TalkError>>>,
    }

    #[derive(Default)]
    struct FakeTransport {
        state: Rc<RefCell<FakeState>>,
    }

    impl Transport for FakeTransport {
        fn start_load(&self, _path: &Path) -> PendingLoad {
            let (tx, rx) = mpsc::channel();
            self.state.borrow_mut().decodes.push(tx);
            PendingLoad::new(rx)
        }
        fn load_at(&self, samples: Vec<f32>, fraction: f64, paused: bool) {
            let mut s = self.state.borrow_mut();
            s.samples = samples.len();
            s.position = (fraction * s.samples as f64) as usize;
            s.paused = paused;
            s.loads += 1;
        }
        fn stop(&self) {
            let mut s = self.state.borrow_mut();
            s.samples = 0;
            s.position = 0;
            s.paused = false;
            s.stops += 1;
        }
        fn pause(&self) {
            self.state.borrow_mut().paused = true;
        }
        fn resume(&self) {
            self.state.borrow_mut().paused = false;
        }
        fn seek(&self, fraction: f64) {
            let mut s = self.state.borrow_mut();
            s.position = (fraction * s.samples as f64) as usize;
        }
        fn progress(&self) -> f64 {
            let s = self.state.borrow();
            if s.samples == 0 {
                0.0
            } else {
                s.position as f64 / s.samples as f64
            }
        }
        fn duration_secs(&self) -> f64 {
            // 1000 samples per second keeps the arithmetic readable.
            self.state.borrow().samples as f64 / 1000.0
        }
        fn is_finished(&self) -> bool {
            let s = self.state.borrow();
            s.samples == 0 || s.position >= s.samples
        }
    }

    #[derive(Default)]
    struct FakeRow {
        idle: RefCell<Vec<bool>>,
        progress: RefCell<Vec<f64>>,
    }

    impl PlaybackRow for FakeRow {
        fn set_idle(&self, finished: bool) {
            self.idle.borrow_mut().push(finished);
        }
        fn set_progress(&self, fraction: f64) {
            self.progress.borrow_mut().push(fraction);
        }
    }

    struct Fixture {
        core: PlaybackCore<FakeTransport>,
        state: Rc<RefCell<FakeState>>,
        row: Rc<FakeRow>,
        row_ref: RowRef,
        t0: Instant,
    }

    fn fixture() -> Fixture {
        let transport = FakeTransport::default();
        let state = Rc::clone(&transport.state);
        let mut core = PlaybackCore::new();
        core.set_transport(transport);
        let row = Rc::new(FakeRow::default());
        let row_ref: RowRef = row.clone();
        Fixture {
            core,
            state,
            row,
            row_ref,
            t0: Instant::now(),
        }
    }

    fn complete_decode(state: &Rc<RefCell<FakeState>>, index: usize, samples: usize) {
        let tx = state.borrow().decodes[index].clone();
        tx.send(Ok(vec![0.5; samples])).expect("decode result");
    }

    #[test]
    fn play_without_a_player_is_refused() {
        let mut core: PlaybackCore<FakeTransport> = PlaybackCore::new();
        let row: RowRef = Rc::new(FakeRow::default());
        assert!(!core.play(Rc::clone(&row), Path::new("a.ogg"), None));
        assert!(!core.is_active(&row));
        assert_eq!(core.tick(Instant::now()), Tick::Idle);
    }

    #[test]
    fn play_loads_the_samples_only_when_the_worker_delivers_them() {
        let mut f = fixture();
        assert!(f.core.play(Rc::clone(&f.row_ref), Path::new("a.ogg"), None));
        assert!(f.core.is_active(&f.row_ref));
        assert!(f.core.is_playing(), "owning the player while loading");
        assert_eq!(f.state.borrow().decodes.len(), 1, "one decode started");
        assert_eq!(f.core.tick(f.t0), Tick::Active);
        assert_eq!(f.state.borrow().loads, 0, "nothing loaded yet");
        complete_decode(&f.state, 0, 4000);
        assert_eq!(f.core.tick(f.t0), Tick::Active);
        assert_eq!(f.state.borrow().loads, 1);
        assert_eq!(f.row.progress.borrow().last(), Some(&0.0));
    }

    #[test]
    fn pause_asked_while_loading_leaves_the_loaded_audio_paused() {
        let mut f = fixture();
        f.core.play(Rc::clone(&f.row_ref), Path::new("a.ogg"), None);
        f.core.pause();
        assert!(f.core.is_paused());
        complete_decode(&f.state, 0, 4000);
        assert_eq!(f.core.tick(f.t0), Tick::Dormant);
        assert!(f.state.borrow().paused, "transport paused after load");
        assert!(
            f.row.progress.borrow().is_empty(),
            "no cursor motion while paused"
        );
        f.core.resume();
        assert!(!f.state.borrow().paused);
        assert_eq!(f.core.tick(f.t0), Tick::Active);
        assert_eq!(f.row.progress.borrow().len(), 1);
    }

    #[test]
    fn seek_asked_while_loading_is_applied_once_loaded() {
        let mut f = fixture();
        f.core
            .play(Rc::clone(&f.row_ref), Path::new("a.ogg"), Some(0.25));
        f.core.seek(0.5);
        assert_eq!(f.state.borrow().position, 0, "nothing to seek in yet");
        complete_decode(&f.state, 0, 4000);
        f.core.tick(f.t0);
        assert_eq!(
            f.state.borrow().position,
            2000,
            "seek applied to the loaded samples"
        );
        assert_eq!(f.row.progress.borrow().last(), Some(&0.5));
    }

    #[test]
    fn a_new_play_supersedes_a_decode_still_running() {
        let mut f = fixture();
        let other: Rc<FakeRow> = Rc::new(FakeRow::default());
        let other_ref: RowRef = other.clone();
        f.core.play(Rc::clone(&f.row_ref), Path::new("a.ogg"), None);
        f.core.play(Rc::clone(&other_ref), Path::new("b.ogg"), None);
        assert_eq!(
            f.row.idle.borrow().as_slice(),
            &[false],
            "first row released at once"
        );
        assert!(f.core.is_active(&other_ref));
        assert!(!f.core.is_active(&f.row_ref));
        // The first decode finishing late changes nothing.
        let stale = f.state.borrow().decodes[0].clone();
        let _ = stale.send(Ok(vec![0.0; 10]));
        assert_eq!(f.core.tick(f.t0), Tick::Active);
        assert_eq!(f.state.borrow().loads, 0, "stale decode discarded");
        complete_decode(&f.state, 1, 2000);
        assert_eq!(f.core.tick(f.t0), Tick::Active);
        assert_eq!(f.state.borrow().loads, 1);
        assert_eq!(other.progress.borrow().len(), 1);
        assert!(f.row.progress.borrow().is_empty());
    }

    #[test]
    fn finished_playback_releases_the_row_and_ends_the_tick() {
        let mut f = fixture();
        f.core.play(Rc::clone(&f.row_ref), Path::new("a.ogg"), None);
        complete_decode(&f.state, 0, 100);
        assert_eq!(f.core.tick(f.t0), Tick::Active);
        f.state.borrow_mut().position = 100;
        assert_eq!(f.core.tick(f.t0), Tick::Idle);
        assert_eq!(f.row.idle.borrow().as_slice(), &[true]);
        assert!(!f.core.is_active(&f.row_ref));
        assert_eq!(f.state.borrow().stops, 2, "samples released at the end");
        assert_eq!(f.core.tick(f.t0), Tick::Idle, "stays idle");
    }

    #[test]
    fn a_failed_decode_releases_the_row() {
        let mut f = fixture();
        f.core.play(Rc::clone(&f.row_ref), Path::new("a.ogg"), None);
        let tx = f.state.borrow().decodes[0].clone();
        tx.send(Err(TalkError::Audio("broken".into())))
            .expect("decode error");
        assert_eq!(f.core.tick(f.t0), Tick::Idle);
        assert_eq!(f.row.idle.borrow().as_slice(), &[false]);
        assert!(!f.core.is_active(&f.row_ref));
    }

    #[test]
    fn stop_releases_the_row_keeping_its_cursor() {
        let mut f = fixture();
        f.core.play(Rc::clone(&f.row_ref), Path::new("a.ogg"), None);
        complete_decode(&f.state, 0, 4000);
        f.core.tick(f.t0);
        f.core.stop();
        assert_eq!(
            f.row.idle.borrow().as_slice(),
            &[false],
            "released, not finished"
        );
        assert!(!f.core.is_active(&f.row_ref));
        assert_eq!(f.state.borrow().samples, 0);
        assert_eq!(f.core.tick(f.t0), Tick::Idle);
    }

    #[test]
    fn the_cursor_is_interpolated_between_transport_updates() {
        let mut f = fixture();
        f.core.play(Rc::clone(&f.row_ref), Path::new("a.ogg"), None);
        complete_decode(&f.state, 0, 10_000); // 10 s at the fake's rate
        f.core.tick(f.t0);
        f.core.tick(f.t0 + Duration::from_secs(1));
        let shown = *f.row.progress.borrow().last().unwrap_or(&-1.0);
        assert!((shown - 0.1).abs() < 1e-6, "1 s into 10 s: {shown}");
        // The transport catches up: the reference point moves with it.
        f.state.borrow_mut().position = 5000;
        f.core.tick(f.t0 + Duration::from_secs(2));
        let shown = *f.row.progress.borrow().last().unwrap_or(&-1.0);
        assert!((shown - 0.5).abs() < 1e-6, "raw jump wins: {shown}");
    }

    #[test]
    fn pause_and_seek_without_an_active_row_do_nothing() {
        let mut f = fixture();
        f.core.pause();
        f.core.seek(0.5);
        f.core.resume();
        assert!(!f.core.is_paused());
        assert!(!f.state.borrow().paused);
        assert_eq!(f.state.borrow().position, 0);
    }

    #[test]
    fn same_path_rebuild_adopts_playback_without_reloading() {
        let mut f = fixture();
        let replacement: RowRef = Rc::new(FakeRow::default());
        assert!(f.core.play(Rc::clone(&f.row_ref), Path::new("a.ogg"), None));
        assert!(f.core.adopt(Rc::clone(&replacement), Path::new("a.ogg")));
        assert!(f.core.is_active(&replacement));
        assert!(!f.core.is_active(&f.row_ref));
        assert_eq!(f.state.borrow().decodes.len(), 1);
        assert!(!f.core.release_if_active(&f.row_ref));
        assert!(f.core.release_if_active(&replacement));
        assert_eq!(f.core.tick(f.t0), Tick::Idle);
    }
}
