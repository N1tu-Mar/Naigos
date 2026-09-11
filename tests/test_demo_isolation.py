"""The import graph between the simulation and the renderer, in one place.

`tests/test_imagery_layers.py` and `tests/test_visual_renderer.py` guard the
vocabulary -- that no module under `naigos/env` or `naigos/rl` so much as names
imagery or a tileset. This file guards the structure underneath that, with the
parser rather than with a regex, because the vocabulary guard is a proxy and
this is the thing it is a proxy for.

Two directions, and they are not symmetric.

**env/rl must not import demo, ever.** The simulation cannot be allowed to
depend on the thing that draws it. If it could, a rendered surface could become
an observation -- which is next-steps E-9 in its most direct form -- and a
change to a viewer could change a result. Nothing rendered may reach a policy.

**demo importing env is expected, and bounded.** `live.py` steps the env; that
is its job. But the modules that own drawing GEOMETRY -- `los.py`, `imagery.py`
-- import nothing from the simulation on purpose, so that "the drawing agrees
with the model" is a claim that can fail rather than one that is true by
construction. That is the same rule `naigos/rl/verifier.py` follows.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "naigos"

#: Modules that draw geometry, and must be checkable against the model rather
#: than derived from it. `live.py` is deliberately NOT here: it owns the env.
INDEPENDENT_RENDERERS = ("demo/los.py", "demo/imagery.py", "demo/models.py", "demo/attitude.py",
                         "demo/modelgen.py", "demo/events.py", "demo/camera.py")


def imported_modules(path: Path) -> set[str]:
    """Every module a file imports, with relative imports resolved to a package.

    AST rather than a line scan: a docstring that explains why a module does not
    import the simulation names the simulation, and must not read as importing
    it. That exact false positive is why this helper exists.
    """
    out: set[str] = set()
    tree = ast.parse(path.read_text())
    # ("naigos", "demo", "los") for a file in the package; a bare filename for
    # anything outside it, which only the teeth-check below passes in.
    rel = path.relative_to(REPO) if path.is_relative_to(REPO) else Path(path.name)
    parts = rel.with_suffix("").parts
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                out.add(node.module or "")
            else:
                # `from ..env.terrain import x` inside naigos/demo/live.py
                base = parts[:len(parts) - node.level]
                out.add(".".join([*base, node.module] if node.module else list(base)))
    return out


def modules_under(rel: str) -> list[Path]:
    return sorted((PKG / rel).rglob("*.py"))


# --- the direction that must never exist ------------------------------------------------


@pytest.mark.parametrize("package", ["env", "rl", "data"])
def test_the_simulation_never_imports_the_renderer(package):
    """A rendered surface must not be able to become an observation, and a change
    to a viewer must not be able to change a result."""
    offenders = []
    for path in modules_under(package):
        for mod in imported_modules(path):
            if mod.split(".")[-2:-1] == ["demo"] or mod.endswith(".demo") or mod == "naigos.demo":
                offenders.append(f"{path.relative_to(REPO)} -> {mod}")
            elif "demo" in mod.split("."):
                offenders.append(f"{path.relative_to(REPO)} -> {mod}")
    assert not offenders, f"the simulation imports the renderer: {offenders}"


@pytest.mark.parametrize("package", ["env", "rl"])
def test_the_simulation_never_imports_a_browser_dependency(package):
    """There is no such dependency to import -- CesiumJS runs in the browser off
    a CDN. This is the check that it stays that way if someone ever adds a
    Python-side viewer library."""
    forbidden = {"cesium", "folium", "pydeck", "plotly", "kaleido"}
    offenders = []
    for path in modules_under(package):
        for mod in imported_modules(path):
            if mod.split(".")[0] in forbidden:
                offenders.append(f"{path.relative_to(REPO)} -> {mod}")
    assert not offenders, f"the simulation imports a rendering library: {offenders}"


def test_the_guard_would_catch_a_real_violation(tmp_path):
    """Proves the parser-based scan has teeth rather than passing vacuously."""
    f = tmp_path / "fake.py"
    f.write_text("from naigos.demo import imagery\n")
    assert "naigos.demo" in imported_modules(f)
    f.write_text('"""Mentions naigos.demo in prose, imports nothing."""\nimport numpy\n')
    assert "naigos.demo" not in imported_modules(f)


# --- the direction that exists, and its boundary ----------------------------------------


@pytest.mark.parametrize("rel", INDEPENDENT_RENDERERS)
def test_the_geometry_modules_do_not_reach_into_the_simulation(rel):
    """`los.py` re-derives the refraction drop rather than calling the env's
    sampler, so `tests/test_los_profile.py` can assert the two agree and mean it.
    An import here would make that assertion tautological."""
    mods = imported_modules(PKG / rel)
    reaching = {m for m in mods
                if m.split(".")[0] in ("jax", "flax", "optax")
                or any(p in ("env", "rl") for p in m.split("."))}
    assert not reaching, f"naigos/{rel} reaches into the simulation: {sorted(reaching)}"


def test_the_server_is_allowed_to_import_the_env():
    """The boundary is a boundary, not a wall. `live.py` steps the simulation --
    that is the whole point of a live viewer -- and this asserts the test above
    is scoped rather than accidentally passing everywhere."""
    mods = imported_modules(PKG / "demo" / "live.py")
    assert any("env" in m.split(".") for m in mods), "live.py should import the env"


# --- and nothing rendered reaches a policy ----------------------------------------------


def test_the_observation_names_no_rendering_concept():
    """The structural check has a direct counterpart: obs.py builds its vectors
    from state and the heightmap, and nothing else is in reach."""
    obs = (PKG / "env" / "obs.py").read_text().lower()
    for word in ("imagery", "tileset", "cesium", "texture", "pixel", "render", "rgb"):
        assert word not in obs, f"naigos/env/obs.py mentions {word!r}"


def test_the_terrain_the_viewer_draws_is_read_not_written():
    """`build_terrain_grid` samples the env's heightmap onto a lat/lon grid. It
    must not be able to hand anything back: the surface belongs to the
    simulation, and the renderer's job is to draw the one it used."""
    src = (PKG / "demo" / "live.py").read_text()
    fn = src[src.index("def build_terrain_grid("):src.index("@dataclass\nclass Counters")]
    assert "sample_height(" in fn
    for mutation in (".at[", "hmap =", "hmap[", "state.hmap ="):
        assert mutation not in fn, f"the terrain endpoint writes to the model: {mutation}"
