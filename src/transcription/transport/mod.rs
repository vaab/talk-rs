//! Transport primitives for transcription API calls.
//!
//! This module contains the resilience and connectivity concerns
//! shared by every provider, across both batch (HTTP POST) and
//! realtime (WebSocket) protocols:
//!
//! - [`http`]: reqwest client configuration, per-request
//!   proportional timeout, progress-reporting request body, model
//!   validation, and model-error enrichment.
//! - [`retry`]: the single retry primitive used by batch HTTP calls
//!   and realtime WebSocket upgrade handshakes.  Every provider's
//!   outgoing API call wraps itself in
//!   [`retry::with_retry`] — there is no "without retry" path.
//! - [`ws`]: shared WebSocket helpers (request builder, ping
//!   keepalive) used by realtime transcribers.

pub(crate) mod http;
pub(crate) mod retry;
pub(crate) mod ws;
