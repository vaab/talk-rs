//! Integration tests for clipboard module.
//!
//! Tests marked with `#[ignore]` require X11 display and xclip installed.
//! Run ignored tests sequentially to avoid clipboard race conditions:
//! `cargo test --test integration_clipboard -- --ignored --test-threads=1`

use talk_rs::clipboard::{Clipboard, MockClipboard, X11Clipboard};

/// Verify that xclip binary is available on the system.
#[tokio::test]
async fn test_xclip_binary_exists() {
    let output = tokio::process::Command::new("which")
        .arg("xclip")
        .output()
        .await
        .expect("failed to run which");

    assert!(
        output.status.success(),
        "xclip is not installed; clipboard tests require xclip"
    );
}

/// Test X11Clipboard set and get roundtrip with real xclip.
/// Requires X11 display and xclip installed.
#[tokio::test]
#[ignore]
async fn test_x11_clipboard_roundtrip() {
    let clipboard = X11Clipboard::new();
    let test_text = "talk-rs clipboard integration test";

    clipboard
        .set_text(test_text)
        .await
        .expect("failed to set clipboard");

    // Small delay to let clipboard settle
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let result = clipboard.get_text().await.expect("failed to get clipboard");

    assert_eq!(result, test_text);
}

/// Test X11Clipboard save and restore pattern with real xclip.
/// Requires X11 display and xclip installed.
#[tokio::test]
#[ignore]
async fn test_x11_clipboard_save_restore() {
    let clipboard = X11Clipboard::new();

    // Set initial content
    clipboard
        .set_text("original content")
        .await
        .expect("failed to set initial clipboard");
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Save current clipboard
    let saved = clipboard
        .get_text()
        .await
        .expect("failed to save clipboard");

    // Set new content (simulating dictate paste)
    clipboard
        .set_text("transcribed text")
        .await
        .expect("failed to set new clipboard");
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let current = clipboard
        .get_text()
        .await
        .expect("failed to get new clipboard");
    assert_eq!(current, "transcribed text");

    // Restore original content
    clipboard
        .set_text(&saved)
        .await
        .expect("failed to restore clipboard");
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let restored = clipboard
        .get_text()
        .await
        .expect("failed to get restored clipboard");
    assert_eq!(restored, "original content");
}

/// Test MockClipboard works correctly in integration context.
#[tokio::test]
async fn test_mock_clipboard_integration() {
    let clipboard = MockClipboard::new();

    // Initially empty
    let text = clipboard
        .get_text()
        .await
        .expect("failed to get empty clipboard");
    assert_eq!(text, "");

    // Set and get
    clipboard
        .set_text("test content")
        .await
        .expect("failed to set clipboard");
    let text = clipboard.get_text().await.expect("failed to get clipboard");
    assert_eq!(text, "test content");
}
