"""Replay is on Cesium's clock, and the clock must not invent anything.

Handing a recording to `viewer.clock` buys play, scrub, rate control and camera
tracking for free. It also buys, by default, a Lagrange fit through the logged
states -- a smooth trajectory the simulation never produced, drawn at the same
fidelity as the ones it did.

So three properties are pinned here, and they are the difference between a
replay and an animation:

  * one sample per logged frame, at the simulation timestep;
  * LINEAR interpolation at degree 1, and no other algorithm anywhere;
  * availability that ends where the track ends, so a lost aircraft stops being
    drawn rather than holding its last position for the rest of the timeline.

Static assertions over the page text, in the style of
`tests/test_visual_renderer.py`: they must pass with no browser, no credential
and no network, which is exactly the condition under which a viewer quietly
starts drawing more than it logged. The Python half checks the data those rules
are applied to -- that `alive` really is a prefix, so "where the track ends" is
a well-defined instant.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
PAGE = (REPO / "naigos" / "demo" / "assets" / "cesium.html").read_text()

#: The replay machinery, isolated. Several assertions below are about what this
#: block does *not* contain, which only means anything if it is really the whole
#: of it.
BLOCK = PAGE[PAGE.index("  if (REPLAY) {"):PAGE.index("  const es = new EventSource(")]


# --- the samples are the logged states --------------------------------------------------


def test_positions_are_a_sampled_property_rather_than_a_per_frame_write():
    assert "new Cesium.SampledPositionProperty()" in BLOCK
    assert "craft[b].position = sp;" in BLOCK


def test_there_is_one_sample_per_logged_frame_at_the_simulation_timestep():
    """`timeAt(f)` is `EPOCH + f * dt`, and `dt` is the recording's own `dt_s` --
    which `naigos.demo.replay.to_json` sets to `cfg.dt * stride`. Nothing here
    picks a display rate and resamples onto it."""
    assert "const dt = rp.dt_s;" in BLOCK
    assert "Cesium.JulianDate.addSeconds(EPOCH, f * dt, new Cesium.JulianDate())" in BLOCK
    assert "sp.addSample(timeAt(f), c3(a.lon, a.lat, a.alt));" in BLOCK
    # one sample per frame index, straight off the frame array
    assert "for (let f = 0; f < nFrames; f++) {" in BLOCK


def test_the_epoch_is_fixed_so_two_exports_of_one_rollout_agree():
    """A recording carries sim seconds, not wall time. Seeding the timeline from
    the clock on the machine would make a committed artifact differ from itself."""
    assert 'Cesium.JulianDate.fromIso8601("2000-01-01T00:00:00Z")' in BLOCK
    assert "Date.now()" not in BLOCK and "new Date(" not in BLOCK


# --- linear, and only linear ------------------------------------------------------------


def test_interpolation_is_linear_at_degree_one():
    """Cesium's default for three or more samples is a Lagrange fit. Between two
    logged states the simulation makes no claim at all, so a straight segment is
    the only thing the data supports; a curve would be invented detail rendered
    at the same fidelity as the measurements."""
    opts = BLOCK[BLOCK.index("sp.setInterpolationOptions({"):
                 BLOCK.index("let last = -1;")]
    assert "interpolationDegree: 1," in opts
    assert "interpolationAlgorithm: Cesium.LinearApproximation," in opts


@pytest.mark.parametrize("algorithm", ["LagrangePolynomialApproximation",
                                       "HermitePolynomialApproximation"])
def test_no_higher_order_interpolation_anywhere_on_the_page(algorithm):
    assert algorithm not in PAGE, f"{algorithm} would smooth the logged states"


def test_the_interpolation_is_set_before_any_sample_is_added():
    """Order matters: options applied after the samples still take effect, but a
    reader cannot tell, and the next edit gets it wrong."""
    assert BLOCK.index("sp.setInterpolationOptions({") < BLOCK.index("sp.addSample(")


# --- availability ends where the track ends ---------------------------------------------


def test_sampling_stops_at_the_step_the_aircraft_was_lost():
    assert "if (!a.alive) break;" in BLOCK


def test_availability_is_the_interval_that_ends_there():
    """Without it Cesium holds the last sample for the rest of the timeline, and
    a shot-down aircraft sits over the map as though it were still flying."""
    assert "craft[b].availability" in BLOCK
    avail = BLOCK[BLOCK.index("craft[b].availability"):BLOCK.index("craft[b].show = true;")]
    assert "new Cesium.TimeInterval({ start: EPOCH, stop: timeAt(last) })" in avail
    # an aircraft with no live frames at all gets an EMPTY collection, not a
    # zero-length interval that Cesium would treat as always-available
    assert "last < 0" in avail
    assert "new Cesium.TimeIntervalCollection()" in avail


def test_the_entity_show_flag_does_not_override_availability():
    """`show` is set once and left; availability is what decides, per instant."""
    assert "craft[b].show = true;" in BLOCK
    # ...and the per-frame writer explicitly stands down in replay
    guard = PAGE[PAGE.index("      if (!isReplay) {"):PAGE.index("      c.point.color =")]
    assert "c.position = pos;" in guard
    assert "c.show = a.alive;" in guard


# --- the widgets, and only where they make sense ----------------------------------------


def test_the_clock_widgets_exist_exactly_in_replay():
    """A live stream has no next frame yet. Scrubbing one would mean inventing
    it, so the animation and timeline widgets are constructed only for a
    recording."""
    assert "animation: REPLAY, timeline: REPLAY, shouldAnimate: REPLAY," in PAGE
    assert 'const REPLAY = scene.mode === "replay";' in PAGE


def test_the_clock_is_configured_rather_than_inherited():
    for setting in ("viewer.clock.startTime", "viewer.clock.stopTime",
                    "viewer.clock.currentTime", "viewer.clock.clockRange",
                    "viewer.clock.clockStep", "viewer.clock.multiplier"):
        assert setting in BLOCK, f"{setting} left at the library default"
    assert "viewer.timeline.zoomTo(" in BLOCK


def test_the_live_path_never_touches_the_clock():
    """The stream drives the page by arrival. If it also drove the clock there
    would be two ideas of "now"."""
    stream = PAGE[PAGE.index("  const es = new EventSource("):]
    assert "viewer.clock" not in stream
    assert "JulianDate" not in stream


# --- the hud, and the camera ------------------------------------------------------------


def test_the_hud_reads_the_nearest_logged_frame():
    """Positions on screen are interpolated between samples; the numbers beside
    them are not, because there is no such thing as an interpolated shootdown."""
    assert "const frameAt = t => Math.min(nFrames - 1, Math.max(0," in BLOCK
    assert "Cesium.JulianDate.secondsDifference(t, EPOCH) / dt))" in BLOCK
    assert "viewer.clock.onTick.addEventListener(c => render(frameAt(c.currentTime)));" in BLOCK


def test_the_camera_tracks_the_entity_rather_than_the_frame():
    """viewer.trackedEntity follows the interpolated position continuously; the
    live path's per-frame lookAt would snap to each logged sample."""
    follow = PAGE[PAGE.index('document.getElementById("t-follow").onclick'):
                  PAGE.index("  // ------------------------------------------------------- visual modes")]
    assert "viewer.trackedEntity = follow" in follow
    assert "c.isAvailable(viewer.clock.currentTime)" in follow


def test_the_track_is_drawn_from_the_sampled_positions():
    """The path graphic samples the same property the aircraft flies, so the
    drawn track cannot drift from the drawn position -- which a second list of
    points could."""
    assert "craft[b].path.resolution = dt;" in BLOCK
    assert "leadTime: 0" in PAGE, "the future must not be drawn before it is played"


def test_a_relief_change_rebuilds_the_tracks():
    """Samples are absolute Cartesians built through c3(), so exaggeration
    invalidates them exactly the way it invalidates the live trails. Missing this
    leaves the terrain rising and the aircraft where they were."""
    assert "if (rebuildTracks) rebuildTracks();" in PAGE
    assert "rebuildTracks = buildTracks;" in BLOCK


def test_the_bespoke_transport_is_gone():
    """Cesium's animation widget owns play and rate; its timeline owns scrub. A
    second set of controls would be a second clock to disagree with."""
    for gone in ('id="rp-play"', 'id="rp-scrub"', "setInterval("):
        assert gone not in PAGE, f"{gone} duplicates a clock widget"
    assert 'id="rp-policies"' in PAGE, "the policy switch is the one thing the widgets cannot do"


# --- the data the rules are applied to --------------------------------------------------


@pytest.fixture(scope="module")
def recording():
    p = REPO / "runs" / "demo" / "demo.json"
    if not p.exists():
        pytest.skip("no recorded rollout; run naigos.demo.replay")
    return json.loads(p.read_text())


def test_alive_is_a_prefix_so_the_track_end_is_a_single_instant(recording):
    """`break` on the first dead frame is only correct if no aircraft comes back.
    The env guarantees it; this is the check that the recording does too."""
    for name, w in recording["worlds"].items():
        alive = np.asarray(w["alive"], dtype=bool)
        for b in range(alive.shape[1]):
            col = alive[:, b]
            if col.all():
                continue
            first_dead = int(np.argmin(col))
            assert not col[first_dead:].any(), f"{name}: aircraft {b} came back from the dead"


def test_the_recording_carries_the_timestep_the_samples_are_spaced_at(recording):
    assert recording["dt_s"] > 0
    n = {len(w["pos"]) for w in recording["worlds"].values()}
    assert len(n) == 1, f"policies have different frame counts: {n}"


def test_the_export_puts_the_timestep_where_the_page_reads_it(tmp_path, recording):
    """The page reads `rp.dt_s` off `/frames` and spaces every sample by it. If
    the export stopped setting it, the samples would pile up at one instant and
    the timeline would collapse -- silently, since the first frame would look
    correct."""
    from naigos.data.geodetic import GeoRef
    from naigos.demo.live import replay_payload

    src = REPO / "runs" / "demo" / "demo.json"
    rp = replay_payload(src, GeoRef(**recording["georef"]), aoi=recording["theatre"])
    assert rp["dt_s"] == pytest.approx(recording["dt_s"])
    assert rp["dt_s"] > 0
    for name in rp["policies"]:
        assert len(rp["frames"][name]) == len(recording["worlds"][name]["pos"])
