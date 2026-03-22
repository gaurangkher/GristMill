//! Lightweight PII scrubber for training buffer writes.
//!
//! Before any query text is inserted into the distillation training buffer, it
//! is run through this scrubber to redact obviously sensitive content.  This is
//! a **hygiene measure**, not a privacy guarantee — it prevents clearly sensitive
//! data from appearing in training records the user may later inspect or export.
//!
//! ## Patterns redacted
//! - Email addresses
//! - US phone numbers (various formats)
//! - US Social Security Numbers (XXX-XX-XXXX)
//! - Credit/debit card numbers (13–19 digit sequences, Luhn-plausible spacing)
//!
//! All matches are replaced with a bracketed label, e.g. `[EMAIL]`, `[PHONE]`,
//! `[SSN]`, `[CARD]`.

use std::sync::OnceLock;

use regex::Regex;

// ─────────────────────────────────────────────────────────────────────────────
// Compiled regex cache (initialised once per process)
// ─────────────────────────────────────────────────────────────────────────────

struct PiiPatterns {
    email: Regex,
    phone: Regex,
    ssn: Regex,
    card: Regex,
}

static PATTERNS: OnceLock<PiiPatterns> = OnceLock::new();

fn patterns() -> &'static PiiPatterns {
    PATTERNS.get_or_init(|| PiiPatterns {
        // Email: local@domain.tld
        email: Regex::new(r"(?i)[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}").expect("email regex"),

        // Phone: +1 (555) 555-5555, 555-555-5555, (555) 555 5555, etc.
        phone: Regex::new(r"(?:\+?1[\s.\-]?)?(?:\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4})")
            .expect("phone regex"),

        // SSN: 123-45-6789 or 123 45 6789
        ssn: Regex::new(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b").expect("ssn regex"),

        // Card: 13-19 digit number with optional spaces/dashes between groups
        // Matches Visa (16), Mastercard (16), Amex (15), Discover (16), etc.
        card: Regex::new(r"\b(?:\d[ \-]?){13,19}\b").expect("card regex"),
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Scrub PII from `text` and return the redacted string.
///
/// Applies patterns in order: card → SSN → phone → email (most-specific first
/// to avoid partial overlaps).  Each match is replaced with a bracketed label.
pub fn scrub(text: &str) -> String {
    let p = patterns();

    // Apply most-specific patterns first to avoid partial overlaps.
    let s = p.card.replace_all(text, "[CARD]");
    let s = p.ssn.replace_all(&s, "[SSN]");
    let s = p.phone.replace_all(&s, "[PHONE]");
    let s = p.email.replace_all(&s, "[EMAIL]");

    s.into_owned()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scrubs_email() {
        let out = scrub("Contact me at alice@example.com please.");
        assert!(
            !out.contains("alice@example.com"),
            "email should be redacted"
        );
        assert!(out.contains("[EMAIL]"), "expected [EMAIL] placeholder");
    }

    #[test]
    fn scrubs_phone_dashes() {
        let out = scrub("Call 555-867-5309 anytime.");
        assert!(!out.contains("555-867-5309"));
        assert!(out.contains("[PHONE]"));
    }

    #[test]
    fn scrubs_phone_parentheses() {
        let out = scrub("Reach me at (800) 555-1234.");
        assert!(!out.contains("555-1234"));
        assert!(out.contains("[PHONE]"));
    }

    #[test]
    fn scrubs_ssn() {
        let out = scrub("My SSN is 123-45-6789.");
        assert!(!out.contains("123-45-6789"));
        assert!(out.contains("[SSN]"));
    }

    #[test]
    fn scrubs_card_number() {
        let out = scrub("Card: 4111 1111 1111 1111 expires next year.");
        assert!(!out.contains("4111"));
        assert!(out.contains("[CARD]"));
    }

    #[test]
    fn preserves_clean_text() {
        let clean = "What is the capital of France?";
        assert_eq!(scrub(clean), clean);
    }

    #[test]
    fn multiple_pii_types_in_one_string() {
        let text = "Email alice@test.com or call 555-123-4567 ref SSN 987-65-4321";
        let out = scrub(text);
        assert!(out.contains("[EMAIL]"));
        assert!(out.contains("[PHONE]"));
        assert!(out.contains("[SSN]"));
        assert!(!out.contains("alice@test.com"));
        assert!(!out.contains("555-123-4567"));
        assert!(!out.contains("987-65-4321"));
    }
}
