"""The visual-mode contract: what the globe is showing, and what it costs to say so.

`--visual` chooses between two postures that differ in one load-bearing way:

  * `physics` draws the simulation's own DEM. What occludes on screen occluded
    in the model, so a screenshot is evidence.
  * `photorealistic` draws Google Photorealistic 3D Tiles, which carry their own
    geometry. The picture is better and it is no longer evidence.

Getting that backwards -- defaulting to the pretty mode, or degrading *into* it
when something is missing -- would re-introduce next-steps E-9 with a nicer
texture on it. So the tests below pin four things:

  1. mode resolution: what you ask for is what you get, and a typo raises;
  2. missing credentials: refuse to guess, never silently upgrade;
  3. safe fallback: every failure path lands on physics/OSM, never the reverse;
  4. token non-persistence: a credential enters `resolve_visual_config` and
     leaves no trace in the object, the page blob or the scene payload.

Static and offline on purpose -- no token, no network, no browser -- which is
exactly the condition under which these rules are easiest to break.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from naigos.demo import imagery

# Dotted like a real ion JWT so the "not even a fragment" check below has a
# payload segment to look for -- but deliberately NOT `eyJ`-prefixed, because
# tests/test_imagery_layers.py scans every tracked file for JWT-shaped strings
# and a realistic fixture would trip the repo's own secret scanner.
TOKEN = "fake-ion-token.not-a-real-payload.0123456789abcdef"
GOOGLE_KEY = "AIzaSy-not-a-real-google-key"


# --- mode resolution --------------------------------------------------------------------


def test_physics_is_the_default_mode():
    """The default must be the evidence-grade one. Everything else is decoration."""
    assert imagery.DEFAULT_VISUAL_MODE == "physics"
    cfg = imagery.resolve_visual_config()
    assert cfg.mode == "physics"
    assert cfg.terrain_source == imagery.TERRAIN_SOURCE_SIMULATION
    assert cfg.evidence_grade is True


def test_physics_with_a_token_takes_the_sentinel2_skin_over_the_simulation_dem():
    cfg = imagery.resolve_visual_config("physics", ion_token=TOKEN)
    assert (cfg.base_imagery, cfg.ion_asset) == ("sentinel2", imagery.SENTINEL2_ION_ASSET)
    assert cfg.tileset is None
    assert cfg.terrain_source == imagery.TERRAIN_SOURCE_SIMULATION
    assert cfg.evidence_grade is True


def test_photorealistic_resolves_through_cesium_ion_when_a_token_is_present():
    cfg = imagery.resolve_visual_config("photorealistic", ion_token=TOKEN)
    assert cfg.mode == "photorealistic"
    assert cfg.tileset == "google_photorealistic"
    assert cfg.tileset_route == "cesium_ion"
    assert cfg.tileset_ion_asset == imagery.GOOGLE_3D_TILES_ION_ASSET


def test_photorealistic_resolves_through_google_when_only_a_maps_key_is_present():
    cfg = imagery.resolve_visual_config("photorealistic", google_api_key=GOOGLE_KEY)
    assert cfg.mode == "photorealistic"
    assert cfg.tileset_route == "google_maps_api"
    # No ion account is involved on this route, so there is no ion asset to name.
    assert cfg.tileset_ion_asset is None


def test_the_ion_route_wins_when_both_credentials_exist():
    """One account, one bill, one place to revoke. Not a correctness rule, but a
    deterministic one -- an ambiguous route makes a support question unanswerable."""
    cfg = imagery.resolve_visual_config(
        "photorealistic", ion_token=TOKEN, google_api_key=GOOGLE_KEY
    )
    assert cfg.tileset_route == "cesium_ion"


def test_photorealistic_is_never_evidence_grade():
    """The whole reason the mode is a mode and not an imagery option."""
    cfg = imagery.resolve_visual_config("photorealistic", ion_token=TOKEN)
    assert cfg.evidence_grade is False
    assert cfg.terrain_source == imagery.TERRAIN_SOURCE_PROVIDER
    assert "PRESENTATION MODE" in cfg.separation_note


def test_an_unknown_visual_mode_raises_rather_than_defaulting():
    """A typo that silently selected physics would be indistinguishable from a
    working run, and the two modes are exactly what must not be confused."""
    with pytest.raises(imagery.VisualConfigError):
        imagery.resolve_visual_config("photoreal")
    with pytest.raises(imagery.VisualConfigError):
        imagery.resolve_visual_config("physics", imagery="bing")


def test_the_mode_that_was_requested_is_recorded_alongside_the_one_in_force():
    cfg = imagery.resolve_visual_config("photorealistic")
    assert (cfg.requested_mode, cfg.mode) == ("photorealistic", "physics")


# --- missing credentials ----------------------------------------------------------------


def test_photorealistic_without_any_credential_falls_back_to_physics():
    cfg = imagery.resolve_visual_config("photorealistic")
    assert cfg.mode == "physics"
    assert cfg.tileset is None
    assert cfg.evidence_grade is True
    assert cfg.terrain_source == imagery.TERRAIN_SOURCE_SIMULATION


def test_the_fallback_names_every_variable_that_would_have_worked():
    """A silent downgrade is a support ticket. The reason has to be actionable."""
    reason = imagery.resolve_visual_config("photorealistic").fallback_reason
    assert reason
    for name in imagery.TOKEN_ENV_VARS + imagery.GOOGLE_API_KEY_ENV_VARS:
        assert name in reason


def test_a_blank_credential_counts_as_missing():
    """An empty `export CESIUM_ION_TOKEN=` left in a shell profile would otherwise
    be sent to the provider and rejected, which surfaces as a blank globe rather
    than as a missing credential."""
    assert imagery.resolve_visual_config("photorealistic", ion_token="   ").mode == "physics"
    assert imagery.resolve_visual_config("photorealistic", google_api_key="").mode == "physics"
    assert imagery.resolve_visual_config("physics", ion_token=" ").base_imagery == "osm"


def test_credentials_are_read_from_named_environment_variables_only():
    assert imagery.resolve_google_api_key({"NAIGOS_GOOGLE_MAPS_API_KEY": "ours"}) == "ours"
    assert imagery.resolve_google_api_key({"GOOGLE_MAPS_API_KEY": "generic"}) == "generic"
    assert imagery.resolve_google_api_key(
        {"NAIGOS_GOOGLE_MAPS_API_KEY": "ours", "GOOGLE_MAPS_API_KEY": "theirs"}
    ) == "ours"
    # Nothing else is a credential source: no dotenv, no config file, no guessing.
    assert imagery.resolve_google_api_key({"GOOGLE_API_KEY": "adjacent"}) is None
    assert imagery.resolve_google_api_key({}) is None


def test_there_is_no_cli_flag_for_the_google_key(capsys):
    """A key passed on argv is a key in the shell history and in every `ps`
    listing on the box. The env var is the only supported path, so the parser
    must reject the flag rather than quietly accept it."""
    from naigos.demo import live

    for flag in ("--google-api-key", "--google-key", "--maps-key"):
        with pytest.raises(SystemExit) as e:
            live.main([flag, GOOGLE_KEY])
        assert e.value.code == 2, f"{flag} was accepted"
    assert "unrecognized arguments" in capsys.readouterr().err


# --- safe fallback ----------------------------------------------------------------------


def test_every_fallback_moves_toward_the_evidence_grade_mode():
    """Degradation has a direction. Nothing may resolve *into* photorealistic.

    urban-presentation is the one mode that does not degrade to physics -- its
    city camera and its disclaimers are the point of asking for it -- so it is
    held to the stricter half of the rule separately: without a credential it
    never resolves into a provider route, and it is never evidence-grade.
    """
    for mode in ("physics", "photorealistic"):
        for kwargs in ({}, {"ion_token": ""}, {"google_api_key": ""}):
            cfg = imagery.resolve_visual_config(mode, **kwargs)
            assert cfg.mode == "physics", f"{mode} {kwargs} did not degrade safely"
            assert cfg.evidence_grade is True
    for kwargs in ({}, {"ion_token": ""}, {"google_api_key": ""}, {"local_urban": True}):
        cfg = imagery.resolve_visual_config("urban-presentation", **kwargs)
        assert cfg.mode == "urban-presentation"
        assert cfg.tileset is None and cfg.tileset_route is None
        assert cfg.evidence_grade is False
        assert cfg.terrain_source == imagery.TERRAIN_SOURCE_SIMULATION


def test_physics_without_a_token_degrades_to_the_keyless_provider_not_to_an_error():
    cfg = imagery.resolve_visual_config("physics")
    assert cfg.base_imagery == "osm"
    assert cfg.ion_asset is None
    assert cfg.osm_url.startswith("https://")
    assert cfg.evidence_grade is True


def test_the_osm_fallback_survives_in_both_modes():
    """OSM is the floor: keyless, and reachable from anywhere in the state space."""
    assert imagery.resolve_visual_config("physics", imagery="osm").base_imagery == "osm"
    # Photorealistic keeps a keyless base layer underneath the tileset, so the
    # globe still has ground where the tileset has no coverage.
    assert imagery.resolve_visual_config(
        "photorealistic", ion_token=TOKEN).base_imagery == "osm"
    assert imagery.resolve_visual_config("photorealistic").base_imagery == "osm"


def test_a_photorealistic_run_never_meters_imagery_nobody_can_see():
    """The tileset is opaque where it has coverage; Sentinel-2 under it would be
    billed against the ion quota and never rendered. Refused, not silently dropped."""
    cfg = imagery.resolve_visual_config("photorealistic", ion_token=TOKEN)
    assert cfg.ion_asset is None
    with pytest.raises(imagery.VisualConfigError):
        imagery.validate_cli("photorealistic", "sentinel2")
    # ...but the flag defaults to None, so `--visual photorealistic` alone works.
    imagery.validate_cli("photorealistic", None)
    imagery.validate_cli("physics", "sentinel2")


def test_the_cli_refuses_an_unknown_mode_before_anything_expensive_starts():
    with pytest.raises(imagery.VisualConfigError):
        imagery.validate_cli("photoreal", None)
    with pytest.raises(imagery.VisualConfigError):
        imagery.validate_cli("physics", "bing")


# --- token non-persistence --------------------------------------------------------------


def _blob(cfg) -> str:
    """Everything the config can be turned into, as one searchable string."""
    return " ".join([
        repr(cfg),
        str(cfg),
        json.dumps(cfg.as_dict()),
        json.dumps(cfg.to_page()),
        json.dumps(dataclasses.asdict(cfg)),
    ])


@pytest.mark.parametrize("mode", imagery.VISUAL_MODES)
def test_no_credential_survives_into_the_config_object(mode):
    """The config is printed at startup, served at /scene and pasted into bug
    reports. A secret in it leaks through all three at once."""
    cfg = imagery.resolve_visual_config(mode, ion_token=TOKEN, google_api_key=GOOGLE_KEY)
    blob = _blob(cfg)
    assert TOKEN not in blob
    assert GOOGLE_KEY not in blob
    # Not even a fragment: a JWT's payload segment alone is enough to be a leak.
    assert TOKEN.split(".")[1] not in blob
    assert "AIzaSy" not in blob


def test_the_config_records_only_whether_a_credential_was_found():
    cfg = imagery.resolve_visual_config("physics", ion_token=TOKEN)
    assert cfg.ion_token_present is True and cfg.google_api_key_present is False
    assert imagery.resolve_visual_config("physics").ion_token_present is False
    fields = {f.name for f in dataclasses.fields(cfg)}
    for f in fields:
        value = getattr(cfg, f)
        assert not isinstance(value, str) or len(value) < 400
    assert not (fields & {"ion_token", "token", "google_api_key", "api_key", "secret"})


def test_the_page_blob_carries_no_credential_field_at_all():
    """The token reaches the page through exactly one substitution point, so
    there is exactly one string in the server to audit."""
    page = imagery.resolve_visual_config(
        "photorealistic", ion_token=TOKEN, google_api_key=GOOGLE_KEY).to_page()
    assert not any("token" in k or "key" in k for k in page)
    assert TOKEN not in json.dumps(page)


def test_the_config_is_frozen_so_the_posture_cannot_change_under_a_screenshot():
    cfg = imagery.resolve_visual_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.evidence_grade = False


def test_resolving_a_config_writes_nothing_to_disk(tmp_path, monkeypatch):
    """Nothing token-shaped is persisted, because nothing at all is persisted."""
    monkeypatch.chdir(tmp_path)
    imagery.resolve_visual_config("photorealistic", ion_token=TOKEN).as_dict()
    imagery.resolve_visual_config("physics", ion_token=TOKEN).to_page()
    assert list(tmp_path.iterdir()) == []


# --- attribution ------------------------------------------------------------------------


def test_every_resolved_mode_carries_an_attribution_for_what_it_draws():
    for cfg in (
        imagery.resolve_visual_config("physics"),
        imagery.resolve_visual_config("physics", ion_token=TOKEN),
        imagery.resolve_visual_config("photorealistic", ion_token=TOKEN),
    ):
        assert cfg.attribution.strip()
        assert cfg.to_page()["attribution"].strip()


def test_the_google_attribution_names_google_and_survives_a_screenshot():
    """Google's terms require the attribution and the per-tile credits on screen.
    Cesium's credit display carries the authoritative version; this string is the
    static restatement, so a still frame carries it too."""
    cfg = imagery.resolve_visual_config("photorealistic", ion_token=TOKEN)
    assert "Google" in cfg.tileset_attribution
    assert "credit display" in cfg.tileset_attribution
    # Both providers are on screen -- the base layer shows where the tileset has
    # no coverage -- so both are credited.
    assert "Google" in cfg.attribution and "OpenStreetMap" in cfg.attribution


def test_the_page_is_credited_for_what_it_actually_draws():
    """The base-layer credit names the provider of the pixels under the cursor.

    The viewer does draw the tileset now, and Google is credited for it: in
    Cesium's credit display, which is never suppressed, and restated in the
    banner. This blob is the base layer alone, which in photorealistic mode is
    OpenStreetMap showing through wherever the tileset has no coverage. A Google
    credit here would sit over pixels Google did not supply, which is the
    opposite of satisfying an attribution requirement."""
    page = imagery.resolve_visual_config("photorealistic", ion_token=TOKEN).to_page()
    assert page["mode"] == "osm"
    assert "Google" not in page["attribution"]


def test_the_google_source_is_allowlisted_with_a_licence_and_attribution_flag():
    from naigos.research.allowlist import ALLOWLIST

    src = ALLOWLIST["google_photorealistic_3d_tiles"]
    assert src.attribution_required
    assert src.license and src.citation and src.role
    # Nothing server-side fetches it; the browser streams the tiles.
    assert src.hosts == ()


# --- the page contract ------------------------------------------------------------------


def test_the_page_has_a_substitution_point_for_every_config_and_credential():
    from pathlib import Path

    page = (Path(__file__).resolve().parents[1]
            / "naigos" / "demo" / "assets" / "cesium.html").read_text()
    assert "/*__VISUAL__*/" in page
    assert "/*__IMAGERY__*/" in page
    assert "/*__ION_TOKEN__*/null" in page
    # Renaming or dropping this one turns render_page's .replace() into a silent
    # no-op: the page keeps `null` and the google_maps_api route fails with a
    # provider error instead of a legible one.
    assert "/*__GOOGLE_API_KEY__*/null" in page


# --- what the served page is allowed to see ---------------------------------------------


@pytest.mark.parametrize(
    "mode,ion,expected_route,key_expected",
    [
        # The default mode never contacts Google, so the key has no business
        # being in a document served to whoever can reach the port.
        ("physics", TOKEN, None, False),
        # Nor does the ion route: CesiumJS reaches the same tileset through ion,
        # which the ion token already covers.
        ("photorealistic", TOKEN, "cesium_ion", False),
        # Only here does the browser talk to Google directly.
        ("photorealistic", None, "google_maps_api", True),
    ],
)
def test_the_google_key_reaches_the_page_only_on_the_route_that_uses_it(
    mode, ion, expected_route, key_expected
):
    """A credential the page cannot use is a credential that should not be in it.

    This is the whole of the server's secret handling in one assertion: render
    the real template, look for the key.
    """
    from naigos.demo import live

    cfg = imagery.resolve_visual_config(mode, ion_token=ion, google_api_key=GOOGLE_KEY)
    assert cfg.tileset_route == expected_route
    html = live.render_page(cfg, ion, GOOGLE_KEY)
    assert (GOOGLE_KEY in html) is key_expected
    # The placeholder is always consumed, whichever way it resolved.
    assert "/*__GOOGLE_API_KEY__*/" not in html


def test_the_ion_token_reaches_the_page_exactly_once_and_only_when_present():
    from naigos.demo import live

    cfg = imagery.resolve_visual_config("physics", ion_token=TOKEN)
    assert live.render_page(cfg, TOKEN).count(TOKEN) == 1
    # No token, nothing token-shaped in the document.
    keyless = imagery.resolve_visual_config("physics")
    html = live.render_page(keyless, None)
    assert TOKEN not in html and "/*__ION_TOKEN__*/" not in html


def test_the_rendered_page_carries_the_config_but_never_a_credential_inside_it():
    """The config blobs are served, printed and logged. A secret in one leaks
    through all three at once, so they must stay boolean-only."""
    from naigos.demo import live

    cfg = imagery.resolve_visual_config(
        "photorealistic", ion_token=TOKEN, google_api_key=GOOGLE_KEY)
    html = live.render_page(cfg, ion_token=TOKEN, google_api_key=GOOGLE_KEY)
    assert json.dumps(cfg.as_dict()) in html
    assert json.dumps(cfg.to_page()) in html
    for blob in (json.dumps(cfg.as_dict()), json.dumps(cfg.to_page())):
        assert TOKEN not in blob and GOOGLE_KEY not in blob


def test_the_served_page_carries_the_config_and_only_one_copy_of_the_token():
    """End-to-end on the substitution itself: the config blobs go in, and the
    token appears exactly where the one substitution point put it."""
    from pathlib import Path

    page = (Path(__file__).resolve().parents[1]
            / "naigos" / "demo" / "assets" / "cesium.html").read_text()
    cfg = imagery.resolve_visual_config("photorealistic", ion_token=TOKEN)
    html = (page
            .replace("/*__ION_TOKEN__*/null", json.dumps(TOKEN))
            .replace('/*__IMAGERY__*/{mode: "osm", osm_url: "https://tile.openstreetmap.org/"}',
                     json.dumps(cfg.to_page()))
            .replace('/*__VISUAL__*/{mode: "physics", evidence_grade: true}',
                     json.dumps(cfg.as_dict())))
    assert html.count(TOKEN) == 1
    assert '"evidence_grade": false' in html
    assert "google_photorealistic" in html


def test_there_is_exactly_one_public_way_to_ask_what_the_globe_is_showing():
    """`imagery_config(token, prefer=...)` was the pre-`--visual` entry point and
    survived the change as a shim over `resolve_visual_config(...).to_page()`.

    Two public answers to one question is a fork waiting to happen: a rule added
    to the resolver -- a credential check, a fallback, a new mode -- is a rule
    the shim's callers silently miss, and the drift shows up as a page config
    that disagrees with the one `/scene` reports. `resolve_visual_config` is the
    single entry point; `to_page()` and `as_dict()` are its two views."""
    assert not hasattr(imagery, "imagery_config"), (
        "the shim is back; callers wanting the base-layer blob want "
        "resolve_visual_config(...).to_page()"
    )
