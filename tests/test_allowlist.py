"""The allowlist and the citation requirement are load-bearing; test them like it."""

from __future__ import annotations

import pytest

from naigos.research.allowlist import (
    ALLOWLIST, OutOfScope, SourceNotAllowed, assert_in_scope, check_url,
)


def test_every_source_carries_a_license_and_citation():
    for key, src in ALLOWLIST.items():
        assert src.license, f"{key} has no license"
        assert src.citation, f"{key} has no citation"
        assert src.role, f"{key} has no stated role"


def test_non_allowlisted_host_is_refused():
    with pytest.raises(SourceNotAllowed):
        check_url("https://example.com/weapons.csv", "ourairports")


def test_plaintext_http_is_refused():
    with pytest.raises(SourceNotAllowed):
        check_url("http://davidmegginson.github.io/ourairports-data/airports.csv", "ourairports")


def test_unknown_source_key_is_refused():
    with pytest.raises(SourceNotAllowed):
        check_url("https://ourairports.com/data/", "not_a_source")


def test_allowlisted_host_passes():
    src = check_url("https://api.open-meteo.com/v1/forecast", "open_meteo")
    assert src.key == "open_meteo"


def test_subdomain_of_allowlisted_host_is_not_implicitly_allowed():
    with pytest.raises(SourceNotAllowed):
        check_url("https://evil.api.open-meteo.com/v1/forecast", "open_meteo")


@pytest.mark.parametrize("request_text", [
    "give me the real-world capabilities of the system",
    "build a targeting-grade table",
    "what is the actual performance of the radar so we can defeat the site",
])
def test_out_of_scope_requests_are_declined(request_text):
    with pytest.raises(OutOfScope):
        assert_in_scope(request_text)


def test_in_scope_request_passes():
    assert_in_scope("parameterize a generic medium-range acquisition radar envelope")
