//! Concrete paste-node implementations.
//!
//! Each module here implements [`crate::paste::PasteNode`] for one
//! variant of [`crate::paste::PasteNodeConfig`].

pub(crate) mod chunk;
pub(crate) mod clipboard;
pub(crate) mod detect;
pub(crate) mod wm_class;
pub(crate) mod xtest;

/// Match a glob pattern (supporting only `*` as a wildcard) against
/// a haystack.  Used by [`wm_class::MatchWmClassNode`].
///
/// Implementation: classic two-pointer with backtracking.  `*`
/// matches any (possibly empty) sequence.  Other characters match
/// literally.  No character classes, no `?`, no escaping — keeping
/// the surface minimal so we don't need an extra dependency.
pub(crate) fn glob_match(pattern: &str, haystack: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let h: Vec<char> = haystack.chars().collect();

    let (mut pi, mut hi) = (0usize, 0usize);
    let (mut star_pi, mut star_hi): (Option<usize>, usize) = (None, 0);

    while hi < h.len() {
        if pi < p.len() && p[pi] == '*' {
            star_pi = Some(pi);
            star_hi = hi;
            pi += 1;
        } else if pi < p.len() && p[pi] == h[hi] {
            pi += 1;
            hi += 1;
        } else if let Some(sp) = star_pi {
            pi = sp + 1;
            star_hi += 1;
            hi = star_hi;
        } else {
            return false;
        }
    }

    // Trailing pattern must be all `*`s.
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

#[cfg(test)]
mod tests {
    use super::glob_match;

    #[test]
    fn glob_literal_match() {
        assert!(glob_match("firefox", "firefox"));
        assert!(!glob_match("firefox", "Firefox"));
    }

    #[test]
    fn glob_star_prefix() {
        assert!(glob_match("*.Emacs", "x.Emacs"));
        assert!(glob_match("*.Emacs", ".Emacs"));
        assert!(!glob_match("*.Emacs", "Emacs"));
    }

    #[test]
    fn glob_star_suffix() {
        assert!(glob_match("firefox.*", "firefox.Firefox"));
        assert!(glob_match("firefox.*", "firefox."));
        assert!(!glob_match("firefox.*", "Firefox.x"));
    }

    #[test]
    fn glob_star_middle() {
        assert!(glob_match("a*z", "az"));
        assert!(glob_match("a*z", "abz"));
        assert!(glob_match("a*z", "abcdefz"));
        assert!(!glob_match("a*z", "ab"));
    }

    #[test]
    fn glob_only_star() {
        assert!(glob_match("*", ""));
        assert!(glob_match("*", "anything"));
    }

    #[test]
    fn glob_empty_pattern() {
        assert!(glob_match("", ""));
        assert!(!glob_match("", "x"));
    }
}
