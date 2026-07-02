//! Compile-time probe that the headless (`--no-default-features`) lib
//! surface exposes everything a cloud-only consumer depends on.
#[allow(unused_imports)]
use talk_rs::audio::{encoder, file_source, mock, resample, writer};
#[allow(unused_imports)]
use talk_rs::transcription::{
    self, jobs, mistral, model_suggestions, openai, openai_realtime, realtime, transport,
};
#[allow(unused_imports)]
use talk_rs::{config, daemon, error, recording_cache, telemetry};
#[allow(unused_imports)]
use talk_rs::{Clipboard, MockClipboard};
#[allow(unused_imports)]
use talk_rs::{
    MistralOneShotTranscriber, MockOneShotTranscriber, OpenAIOneShotTranscriber,
    OpenAIRealtimeTranscriber,
};

fn main() {}
