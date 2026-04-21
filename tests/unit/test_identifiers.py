"""Canonicaliser tests. Each function must return a stable canonical form
for a valid input and None for anything malformed — conflict detection
relies on this."""

from src.consolidator import identifiers as I


# ── canon_lei ────────────────────────────────────────────────────────

def test_lei_uppercase_20_chars_accepted():
    assert I.canon_lei("529900WTOG7RHO5TCH58") == "529900WTOG7RHO5TCH58"


def test_lei_lowercase_uppercased():
    assert I.canon_lei("529900wtog7rho5tch58") == "529900WTOG7RHO5TCH58"


def test_lei_too_short_rejected():
    assert I.canon_lei("529900") is None


def test_lei_empty_and_none():
    assert I.canon_lei("") is None
    assert I.canon_lei(None) is None


# ── canon_cik ────────────────────────────────────────────────────────

def test_cik_strips_leading_zeros():
    assert I.canon_cik("0000000123") == "123"
    assert I.canon_cik("123") == "123"


def test_cik_zero_canonicalises_to_zero():
    assert I.canon_cik("0") == "0"
    assert I.canon_cik("0000") == "0"


def test_cik_nondigit_rejected():
    assert I.canon_cik("ABC") is None
    assert I.canon_cik("12X34") is None


# ── canon_vat ────────────────────────────────────────────────────────

def test_vat_de_valid():
    assert I.canon_vat("DE273691032") == "DE273691032"


def test_vat_de_with_spaces_normalised():
    assert I.canon_vat("DE 273 691 032") == "DE273691032"
    assert I.canon_vat("DE-273.691.032") == "DE273691032"


def test_vat_fr_valid():
    assert I.canon_vat("FR12345678901") == "FR12345678901"


def test_vat_de_wrong_length_rejected():
    assert I.canon_vat("05513000") is None       # too short, no country prefix
    assert I.canon_vat("DE12345") is None         # too short


def test_vat_ted_notice_rejected():
    """TED notices wrongly written to `vat` must not be treated as VAT."""
    assert I.canon_vat("1594225-1-0-1") is None
    assert I.canon_vat("83409669500855-1-2-3") is None


def test_vat_siret_alone_rejected_as_vat():
    """A bare SIRET is not a VAT — France's VAT starts with FR."""
    assert I.canon_vat("83415751300815") is None


def test_vat_empty_and_none():
    assert I.canon_vat("") is None
    assert I.canon_vat(None) is None


# ── canon_siret ──────────────────────────────────────────────────────

def test_siret_14_digits_accepted():
    assert I.canon_siret("83415751300815") == "83415751300815"


def test_siret_with_spaces_normalised():
    assert I.canon_siret("834 096 745 00197") == "83409674500197"


def test_siret_wrong_length_rejected():
    assert I.canon_siret("123") is None
    assert I.canon_siret("834157513008151234") is None


# ── looks_like_ted_notice ───────────────────────────────────────────

def test_ted_notice_pattern():
    assert I.looks_like_ted_notice("1594225-1-0-1") is True
    assert I.looks_like_ted_notice("1-2-3-4") is True


def test_not_ted_notice():
    assert I.looks_like_ted_notice("DE273691032") is False
    assert I.looks_like_ted_notice("83415751300815") is False
    assert I.looks_like_ted_notice(None) is False
