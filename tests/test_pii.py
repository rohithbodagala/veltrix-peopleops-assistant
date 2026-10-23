"""Personal data handling (spec 7.4)."""
import logging

from app.logging_setup import RedactingJsonFormatter
from app.pii import NAMES, mask_emails, scrub_answer, scrub_value, strip_phones


def test_emails_are_masked_first_char_then_domain():
    assert mask_emails("Write to asha.kumar@veltrix.example today") == "Write to a***@veltrix.example today"
    assert mask_emails("hr@veltrix.example, it-help@veltrix.example") == "h***@veltrix.example, i***@veltrix.example"
    assert mask_emails("a***@veltrix.example") == "a***@veltrix.example"  # idempotent


def test_phone_numbers_are_removed():
    for phone in ("+91 98221 51875", "+1 512-555-0182", "+49 30 5550 2681", "(512) 555-0182", "512-555-0182"):
        assert "555" not in strip_phones(f"call {phone} now") and "98221" not in strip_phones(f"call {phone}")


def test_phone_regex_ignores_dates_ids_uuids_and_amounts():
    text = ("Leave 2026-11-02 to 2026-11-06 for E2004, request 66160227-1737-4472-ac16-72297608d942, "
            "band 6,00,000 to 9,00,000, 12 days")
    assert strip_phones(text) == text


def test_scrub_value_walks_tool_args():
    args = {"note": "mail x.y@veltrix.example", "nested": ["+91 98221 51875"], "n": 3}
    out = scrub_value(args)
    assert out["note"] == "mail x***@veltrix.example" and "98221" not in out["nested"][0] and out["n"] == 3


def test_answers_are_scrubbed():
    assert scrub_answer("Contact b.ford@veltrix.example or +1 512-555-0140.") == \
        "Contact b***@veltrix.example or [phone number removed]."


def test_log_lines_replace_names_with_ids_and_mask_contacts():
    NAMES.load([("E2002", "Asha Kumar"), ("E3002", "Ben Ford")])
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "x", (), None)
    record.fields = {"event": "debug", "text": "Asha Kumar asked about Ben; mail asha.kumar@veltrix.example, +91 98000 00001"}
    line = RedactingJsonFormatter().format(record)
    assert "Asha" not in line and "Kumar" not in line and "Ben" not in line
    assert "E2002" in line and "E3002" in line
    assert "asha.kumar@" not in line and "98000" not in line
