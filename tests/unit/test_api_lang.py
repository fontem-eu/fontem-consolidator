"""Tests for src/api/lang.py — safe_lang + apply_translation."""
from __future__ import annotations

import pytest

from src.api.lang import EU_LANGS, apply_translation, safe_lang


class TestSafeLang:
    def test_accepts_each_eu_code(self):
        for c in EU_LANGS:
            assert safe_lang(c) == c

    @pytest.mark.parametrize("raw,expected", [
        ("EN", "en"),
        (" de ", "de"),
        ("pt-BR", "pt"),
        ("en_GB", "en"),
    ])
    def test_normalises(self, raw, expected):
        assert safe_lang(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "ja", "zh-CN", "x", "<script>", 123, object()])
    def test_rejects(self, raw):
        assert safe_lang(raw) is None


class TestApplyTranslation:
    def test_noop_when_lang_is_none(self):
        props = {"name": "Ministero", "country": "ITA", "name_de": "Ministerium"}
        assert apply_translation(props, None) == props
        # Returns the same dict (no copy needed)
        assert apply_translation(props, None) is props

    def test_swaps_name_when_translation_present(self):
        props = {"name": "Ministero", "country": "ITA", "name_de": "Ministerium"}
        out = apply_translation(props, "de")
        assert out["name"] == "Ministerium"
        assert out["country"] == "ITA"
        # Originals preserved alongside
        assert out["name_de"] == "Ministerium"

    def test_does_not_mutate_input(self):
        props = {"name": "Ministero", "name_de": "Ministerium"}
        _ = apply_translation(props, "de")
        assert props["name"] == "Ministero"

    def test_falls_through_when_translation_missing(self):
        # Italian authority has no name_fr — should return original unchanged
        props = {"name": "Ministero", "country": "ITA"}
        out = apply_translation(props, "fr")
        assert out == props

    def test_falls_through_when_translation_empty(self):
        props = {"name": "Ministero", "name_fr": ""}
        out = apply_translation(props, "fr")
        assert out["name"] == "Ministero"

    def test_company_props_untouched(self):
        # Company nodes don't carry name_<lang> — coalesce is a no-op
        props = {"name": "BOSCH GMBH", "gmr_id": "abc", "vat": "DE123"}
        out = apply_translation(props, "fr")
        assert out == props
