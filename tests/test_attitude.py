"""The orientation conversion, checked against CesiumJS's rotation re-derived here.

`naigos.demo.attitude` turns the simulation's (psi, gamma, phi) into the three
angles the page hands `Cesium.HeadingPitchRoll`. A test that called the module
to check the module would pass whatever the signs were, so this file carries its
own copy of what CesiumJS does with those angles -- `Quaternion.fromHeadingPitchRoll`
(R = Rz(-h) Ry(-p) Rx(r)) and `ModelUtility.getAxisCorrectionMatrix` for a glTF
2.0 file (Y_UP_TO_Z_UP . Z_UP_TO_X_UP) -- read off the CesiumJS 1.145 source,
and asks where the model's nose and wings end up in east-north-up.

The claims pinned: the nose points along the simulated velocity, including its
climb; a banked aircraft shows the bank on the side the dynamics say it turns
to; and a turret points at the aircraft it is tracking.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from naigos.demo import attitude
from naigos.env import airframe
from naigos.env.config import AirframeConfig


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


#: CesiumJS Axis.Y_UP_TO_Z_UP and Axis.Z_UP_TO_X_UP, from their column-major
#: Matrix3.fromArray literals.
Y_UP_TO_Z_UP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).T
Z_UP_TO_X_UP = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]).T
GLTF_CORRECTION = Y_UP_TO_Z_UP @ Z_UP_TO_X_UP


def cesium_axes(h_deg, p_deg, r_deg):
    """Where a glTF model's forward (+Z), left (+X) and up (+Y) point, in ENU."""
    h, p, r = map(math.radians, (h_deg, p_deg, r_deg))
    R = _rz(-h) @ _ry(-p) @ _rx(r) @ GLTF_CORRECTION
    return R @ [0, 0, 1], R @ [1, 0, 0], R @ [0, 1, 0]


def frame_axes(psi, gamma, phi):
    """The whole server->page chain on the grid (no convergence)."""
    hpr = attitude.cesium_hpr_deg(attitude.grid_heading_deg(psi),
                                  attitude.pitch_deg(gamma), attitude.roll_deg(phi))
    return cesium_axes(*hpr)


# --- the re-derivation itself ---------------------------------------------------------


def test_the_axis_correction_is_the_one_cesium_documents():
    """glTF +Z (forward) -> Cesium model +X; +Y (up) -> +Z; +X (left) -> +Y."""
    assert np.allclose(GLTF_CORRECTION @ [0, 0, 1], [1, 0, 0])
    assert np.allclose(GLTF_CORRECTION @ [0, 1, 0], [0, 0, 1])
    assert np.allclose(GLTF_CORRECTION @ [1, 0, 0], [0, 1, 0])


# --- heading ---------------------------------------------------------------------------


@pytest.mark.parametrize("psi_deg,enu", [(0, (1, 0)), (90, (0, 1)), (180, (-1, 0)),
                                         (-90, (0, -1)), (45, (0.7071, 0.7071))])
def test_the_nose_points_along_the_simulated_heading(psi_deg, enu):
    nose, _, up = frame_axes(math.radians(psi_deg), 0.0, 0.0)
    assert nose[:2] == pytest.approx(enu, abs=1e-4)
    assert nose[2] == pytest.approx(0.0, abs=1e-9)
    assert up == pytest.approx([0, 0, 1], abs=1e-9)


def test_compass_heading_is_clockwise_from_north():
    assert attitude.grid_heading_deg(0.0) == pytest.approx(90.0)          # east
    assert attitude.grid_heading_deg(math.pi / 2) == pytest.approx(0.0)   # north
    assert attitude.grid_heading_deg(-math.pi / 2) == pytest.approx(180.0)
    assert 0.0 <= float(attitude.grid_heading_deg(-0.1)) < 360.0


# --- climb -----------------------------------------------------------------------------


@pytest.mark.parametrize("gamma", [0.3, -0.25])
def test_a_climbing_aircraft_is_drawn_climbing(gamma):
    """Not level: the nose carries the flight-path angle, which is exactly what
    inferring attitude from a horizontal velocity would erase."""
    psi = 0.7
    nose, _, _ = frame_axes(psi, gamma, 0.0)
    assert nose[2] == pytest.approx(math.sin(gamma), abs=1e-9)
    assert math.atan2(nose[1], nose[0]) == pytest.approx(psi, abs=1e-9)


def test_the_nose_is_the_simulated_velocity_vector():
    """The point-mass airframe has no angle of attack: its velocity IS its nose.
    Checked against airframe.velocity(), the function the env itself uses."""
    rng = np.random.default_rng(0)
    for _ in range(25):
        psi, gamma, phi = rng.uniform(-3, 3), rng.uniform(-0.35, 0.35), rng.uniform(-1.4, 1.4)
        st = airframe.AircraftState(pos=jnp.zeros(3), speed=jnp.float32(180.0), psi=jnp.float32(psi),
                                    gamma=jnp.float32(gamma), phi=jnp.float32(phi),
                                    fuel=jnp.float32(1.0))
        v = np.asarray(airframe.velocity(st), dtype=np.float64)
        nose, _, _ = frame_axes(psi, gamma, phi)
        assert nose == pytest.approx(v / np.linalg.norm(v), abs=1e-5)


# --- bank ------------------------------------------------------------------------------


def test_the_bank_is_on_the_side_the_dynamics_turn_to():
    """Positive bank action -> positive phi -> psi grows (counter-clockwise, a
    LEFT turn) under the env's own coordinated-turn integrator. The drawn model
    must then have its LEFT wing down. Getting the sign wrong draws every turn as
    a skid the other way."""
    cfg = AirframeConfig()
    st = airframe.AircraftState(pos=jnp.zeros(3), speed=jnp.float32(180.0), psi=jnp.float32(0.0),
                                gamma=jnp.float32(0.0), phi=jnp.float32(0.0), fuel=jnp.float32(1000.0))
    for _ in range(6):
        st = airframe.step(cfg, st, jnp.array([0.8, 0.0, 0.2]), 0.5)
    phi, psi = float(st.phi), float(st.psi)
    assert phi > 0.3 and psi > 0.0, "the env turns left under positive bank"
    assert float(attitude.roll_deg(phi)) < 0, "a left bank is negative right-wing-down roll"

    nose, left, up = frame_axes(psi, 0.0, phi)
    assert left[2] < -0.3, "left wing down"
    assert up[2] > 0.0
    # and the lift vector leans into the turn: toward the left of the nose
    left_of_nose = np.cross([0, 0, 1], nose)
    assert np.dot(up, left_of_nose) > 0.3


def test_bank_and_climb_survive_together():
    nose, left, _ = frame_axes(1.1, 0.2, -0.6)      # climbing right turn
    assert nose[2] == pytest.approx(math.sin(0.2), abs=1e-9)
    assert left[2] > 0.3, "right bank: left wing up"


def test_zero_state_is_level_wings():
    _, left, up = frame_axes(0.3, 0.0, 0.0)
    assert left[2] == pytest.approx(0.0, abs=1e-12)
    assert up == pytest.approx([0, 0, 1], abs=1e-12)


# --- grid convergence ------------------------------------------------------------------


def test_true_heading_removes_grid_convergence():
    """UTM grid north is not true north. The corrected heading makes the drawn
    nose follow the ground track the georeferenced positions trace."""
    pytest.importorskip("pyproj")
    from naigos.data.geodetic import GeoRef

    # Tehran's zone, well east of its central meridian so convergence is real.
    g = GeoRef(utm_epsg=32639, origin_easting_m=620_000.0, origin_northing_m=3_950_000.0)
    x, y = np.array([5_000.0, 40_000.0]), np.array([5_000.0, 60_000.0])
    psi = np.array([0.4, 2.0])
    true = attitude.true_heading_deg(g.to_wgs84, x, y, psi)
    grid = attitude.grid_heading_deg(psi)
    conv = (true - grid + 180) % 360 - 180
    assert 0.2 < abs(conv[0]) < 3.0, "convergence exists here and is small"
    # the positions one second of flight apart, projected, bear the true heading
    lon0, lat0 = g.to_wgs84(x, y)
    lon1, lat1 = g.to_wgs84(x + 180 * np.cos(psi), y + 180 * np.sin(psi))
    assert attitude.bearing_deg(lon0, lat0, lon1, lat1) == pytest.approx(true, abs=0.01)


def test_bearing_is_compass_convention():
    assert attitude.bearing_deg(51.0, 35.0, 51.0, 35.1) == pytest.approx(0.0, abs=1e-9)
    assert attitude.bearing_deg(51.0, 35.0, 51.1, 35.0) == pytest.approx(90.0, abs=0.1)
    assert attitude.bearing_deg(51.0, 35.0, 50.9, 35.0) == pytest.approx(270.0, abs=0.1)


# --- turret ----------------------------------------------------------------------------


@pytest.mark.parametrize("hull,target", [(0, 90), (90, 0), (200, 170), (350, 10), (45, 45), (10, 190)])
def test_the_turret_points_at_the_bearing_it_was_given(hull, target):
    """Node rotation about glTF +Y, composed with the hull's own HPR, puts the
    turret's forward on the target bearing."""
    yaw = math.radians(float(attitude.sensor_yaw_deg(hull, target)))
    turret_fwd_gltf = _ry(yaw) @ [0, 0, 1]
    h, p, r = attitude.cesium_hpr_deg(hull, 0.0, 0.0)
    R = _rz(-math.radians(h)) @ _ry(-math.radians(p)) @ _rx(math.radians(r)) @ GLTF_CORRECTION
    fwd = R @ turret_fwd_gltf
    compass = math.degrees(math.atan2(fwd[0], fwd[1])) % 360
    assert (compass - target + 180) % 360 - 180 == pytest.approx(0.0, abs=1e-6)


def test_the_turret_yaw_is_wrapped():
    for hull in range(0, 360, 37):
        for tgt in range(0, 360, 41):
            yaw = float(attitude.sensor_yaw_deg(hull, tgt))
            assert -180.0 < yaw <= 180.0


def test_the_module_imports_nothing_from_the_simulation():
    from pathlib import Path

    import ast

    tree = ast.parse(Path(attitude.__file__).read_text())
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    mods |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert mods <= {"__future__", "numpy"}, mods
    assert jax  # the test may use the env; the module may not
