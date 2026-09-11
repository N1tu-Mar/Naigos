"""Aircraft and threat orientation: simulation state -> what CesiumJS rotates by.

The page used to draw every aircraft as a point, so orientation never had to be
right. A model has a nose and wings, and every one of them is a claim: which way
the aircraft was going, whether it was climbing, which wing was down. So this
module converts the simulation's own attitude state, explicitly, and
`tests/test_attitude.py` checks the conversion against an independent
re-derivation of CesiumJS's rotation rather than against itself.

Conventions, all three of them written down:

**Simulation** (`naigos.env.airframe`, local ENU metres on a UTM grid):

    psi    heading, rad, 0 = grid east (+x), counter-clockwise positive
    gamma  flight-path angle, rad, positive = climbing
    phi    bank, rad, sign fixed by the coordinated turn psi_dot = g tan(phi) / V:
           positive phi turns psi counter-clockwise, i.e. a LEFT turn, so
           positive phi is LEFT wing down

**Frame fields** (what `/stream` and `/frames` carry; aviation-style degrees):

    heading  compass degrees from TRUE north, clockwise, in [0, 360)
    pitch    degrees, positive nose up           = degrees(gamma)
    roll     degrees, positive RIGHT wing down   = -degrees(phi)

The point-mass airframe has no angle of attack or sideslip, so pitch is the
flight-path angle and the nose points along the velocity vector. That is the
simulation's own model, not an approximation added here -- and it is why this
module never derives attitude from successive positions: a finite difference
would erase bank entirely and lag every climb by a frame.

**CesiumJS** (`Cesium.HeadingPitchRoll` in a local east-north-up frame):

    heading  about -z, 0 = model +X along local east, positive toward south
    pitch    about -y, positive nose up
    roll     about +x, positive lifts the left side (right wing down)
    order    R = Rz(-h) . Ry(-p) . Rx(r)      (roll applied first)

and CesiumJS's default glTF 2.0 axis correction puts a file's +Z (forward) on
that +X and its +Y (up) on +Z. So for a model authored to the glTF convention:

    cesium_heading = compass_heading - 90
    cesium_pitch   = pitch
    cesium_roll    = roll

plus the registry's per-model offsets (all zero for the shipped set).

Grid versus true north. psi is measured on the UTM grid, and UTM grid north
differs from true north by the meridian convergence -- about 0.5 degrees across
the Tehran AOI, more near a zone edge. `true_heading_deg` removes it by
projecting a short step along psi back to WGS84 and taking the geodesic bearing,
so the nose points along the same ground track the positions trace.

Numpy only; imports nothing from the simulation.
"""

from __future__ import annotations

import numpy as np

#: Length of the step used to measure heading on the ellipsoid. Short enough
#: that convergence is constant over it, long enough to stay far above the
#: 1e-6-degree rounding of the frame coordinates.
HEADING_PROBE_M = 100.0


def bearing_deg(lon0, lat0, lon1, lat1):
    """Initial great-circle bearing, degrees clockwise from true north, [0, 360)."""
    p0, p1 = np.radians(lat0), np.radians(lat1)
    dl = np.radians(np.asarray(lon1) - np.asarray(lon0))
    y = np.sin(dl) * np.cos(p1)
    x = np.cos(p0) * np.sin(p1) - np.sin(p0) * np.cos(p1) * np.cos(dl)
    return np.degrees(np.arctan2(y, x)) % 360.0


def grid_heading_deg(psi):
    """Compass heading on the grid (no convergence correction), [0, 360)."""
    return (90.0 - np.degrees(psi)) % 360.0


def true_heading_deg(to_wgs84, x, y, psi):
    """Compass heading from TRUE north for a platform at grid (x, y) heading psi.

    `to_wgs84(x, y) -> (lon, lat)` is the scene's own georef transform, passed
    in so this module needs no projection library of its own.
    """
    x, y, psi = (np.asarray(v, dtype=np.float64) for v in (x, y, psi))
    lon0, lat0 = to_wgs84(x, y)
    lon1, lat1 = to_wgs84(x + HEADING_PROBE_M * np.cos(psi), y + HEADING_PROBE_M * np.sin(psi))
    return bearing_deg(lon0, lat0, lon1, lat1)


def pitch_deg(gamma):
    return np.degrees(gamma)


def roll_deg(phi):
    """Right-wing-down positive, from the simulation's left-wing-down-positive phi."""
    return -np.degrees(phi)


def cesium_hpr_deg(heading, pitch, roll, spec=None):
    """Frame attitude -> the three numbers the page hands `HeadingPitchRoll.fromDegrees`.

    Mirrors `modelHpr()` in `assets/cesium.html` exactly; `spec` is a
    `naigos.demo.models.ModelSpec` (or its page dict) supplying axis offsets.
    """
    def off(name):
        if spec is None:
            return 0.0
        return float(spec[name] if isinstance(spec, dict) else getattr(spec, name))

    return (heading - 90.0 + off("heading_offset_deg"),
            pitch + off("pitch_offset_deg"),
            roll + off("roll_offset_deg"))


def sensor_yaw_deg(platform_heading, bearing_to_target):
    """Rotation of a turret/sensor node about its local +Y, degrees, in (-180, 180].

    glTF +Y rotation turns the node's forward (+Z) toward +X, which is the
    model's LEFT. A target clockwise of the platform heading (to its right)
    therefore needs a negative angle.
    """
    rel = (np.asarray(bearing_to_target) - np.asarray(platform_heading) + 180.0) % 360.0 - 180.0
    return -rel
