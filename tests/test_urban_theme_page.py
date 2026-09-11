"""The local city layer's theme: colour for local OSM extrusions, and only for them.

Offline: no network, no browser, no GPU. What is under test:

* the gate -- the theme is applied to the local OSM extrusions and to nothing
  else; the provider path (Google Photorealistic 3D Tiles) never reads it;
* the palette -- a pure function of the footprint record and the theatre, so
  one cache always paints the same city whatever order its chunks load in;
  theatre-aware; vibrant but restrained; clear of the overlay colours;
* the roads -- every class the builder can emit has a width and a colour,
  widest first, and stronger than the layer it replaces;
* the roofs -- the shader patch hits CesiumJS's own anchors or steps aside;
* the contract -- presentation-only, terrain-only draping, see-through by
  default with every marker on top, and no new network or raster asset.

The pure theme block (``/*__URBAN_THEME_BEGIN__*/`` ... ``END``) runs under node
when node is installed.
"""

from __future__ import annotations

import colorsys
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from naigos.demo import cities, urban

REPO = Path(__file__).resolve().parents[1]
PAGE = (REPO / "naigos" / "demo" / "assets" / "cesium.html").read_text()
THEME_JS = PAGE[PAGE.index("/*__URBAN_THEME_BEGIN__*/"):PAGE.index("/*__URBAN_THEME_END__*/")]
URBAN_JS = PAGE[PAGE.index("/*__URBAN_PURE_BEGIN__*/"):PAGE.index("/*__URBAN_PURE_END__*/")]
CITY_BLOCK = PAGE[PAGE.index("// --------------------------------------------------------------- city layer"):
                  PAGE.index("// ------------------------------------------------------- visual modes")]
BUILD_CHUNK = PAGE[PAGE.index("function buildChunk(ci)"):PAGE.index("function showChunk(rec)")]
PHOTO = PAGE[PAGE.index("async function showPhotorealisticContext()"):PAGE.index("photoBtn.onclick =")]

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node not installed")

#: What the layer looked like before the theme: one colour, four thin roads.
OLD_BUILDING = "#d9cfbd"
OLD_ROAD_WIDTHS = [4.0, 2.6, 1.6, 1.1]
OLD_ROAD_ALPHA = [0.85, 0.75, 0.6, 0.5]
#: The page's overlay colours (threat red, detection green, HUD amber).
OVERLAYS = {"red": "#ff5a5a", "green": "#45d17a"}
#: Every road class the builder can emit.
ROAD_CLASSES = sorted(set(urban.ROAD_CLASSES.values()) | set(urban.MINOR_ROAD_CLASSES.values()))

# CesiumJS 1.145's PerInstanceColorAppearance shaders (@cesium/engine 26.3.0),
# verbatim: the text the roof patch is anchored to.
STOCK_VS = """in vec3 position3DHigh;
in vec3 position3DLow;
in vec3 normal;
in vec4 color;
in float batchId;

out vec3 v_positionEC;
out vec3 v_normalEC;
out vec4 v_color;

void main()
{
    vec4 p = czm_computePosition();

    v_positionEC = (czm_modelViewRelativeToEye * p).xyz;      // position in eye coordinates
    v_normalEC = czm_normal * normal;                         // normal in eye coordinates
    v_color = color;

    gl_Position = czm_modelViewProjectionRelativeToEye * p;
}
"""
STOCK_FS = """in vec3 v_positionEC;
in vec3 v_normalEC;
in vec4 v_color;

void main()
{
    vec3 positionToEyeEC = -v_positionEC;

    vec3 normalEC = normalize(v_normalEC);
#ifdef FACE_FORWARD
    normalEC = faceforward(normalEC, vec3(0.0, 0.0, 1.0), -normalEC);
#endif

    vec4 color = czm_gammaCorrect(v_color);

    czm_materialInput materialInput;
    materialInput.normalEC = normalEC;
    materialInput.positionToEyeEC = positionToEyeEC;
    czm_material material = czm_getDefaultMaterial(materialInput);
    material.diffuse = color.rgb;
    material.alpha = color.a;

    out_FragColor = czm_phong(normalize(positionToEyeEC), material, czm_lightDirectionEC);
}
"""


def _node(expr: str):
    """Evaluate ``expr`` after the page's two pure urban blocks, under node."""
    script = URBAN_JS + THEME_JS + f"\nprocess.stdout.write(JSON.stringify(({expr})));\n"
    out = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def _rgb(hex_: str) -> tuple[float, float, float]:
    n = int(hex_[1:], 16)
    return ((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255


def _hsv(hex_: str) -> tuple[float, float, float]:
    return colorsys.rgb_to_hsv(*_rgb(hex_))


def _way(i, pts, tags=None):
    ring = [*pts, pts[0]]
    return {"type": "way", "id": i, "tags": tags or {"building": "yes"},
            "geometry": [{"lon": x, "lat": y} for x, y in ring]}


@pytest.fixture(scope="module")
def payload():
    """A small derived city: many footprints, low-rise and towers, two chunks."""
    bounds = urban.UrbanBounds("t", 10.0, 20.0, 10.1, 20.1, "test")
    els = []
    for k in range(240):
        lon, lat = 10.001 + 0.0004 * (k % 20), 20.001 + 0.0004 * (k // 20)
        if k % 2:
            lon += 0.05     # every other one in a second chunk
        d = 0.00012 + 0.00001 * (k % 7)
        tags = {"building": "yes", "height": str(8 + (k * 7) % 90)}
        els.append(_way(k + 1, [(lon, lat), (lon + d, lat), (lon + d, lat + d), (lon, lat + d)], tags))
    p = urban.derive({"elements": els}, bounds, {})
    assert p["counts"]["buildings"] == 240 and len(p["chunks"]) >= 2
    return p


# --- the pure block ----------------------------------------------------------------------


def test_the_theme_block_is_pure_and_offline():
    for word in ("Cesium.", "document.", "window.", "viewer", "fetch(", "XMLHttpRequest",
                 "Math.random", "Date", "performance.", "http", "url(", ".png", ".jpg"):
        assert word not in THEME_JS, word


def test_every_shipped_city_has_its_own_theme():
    themes = re.findall(r"^  (\w+): \{$", THEME_JS[THEME_JS.index("const URBAN_THEMES"):], re.M)
    assert "default" in themes
    for aoi in cities.CITIES:
        assert aoi in themes, f"{aoi} has a city layer but no palette"


@needs_node
def test_the_theme_follows_the_payload_then_the_theatre_then_the_default():
    r = _node("""[urbanThemeKey("dubai_urban", "tehran_basin"), urbanThemeKey(null, "mecca_urban"),
                   urbanThemeKey(undefined, undefined), urbanTheme("owens_valley") === URBAN_THEMES.default,
                   urbanTheme("dubai_urban").name]""")
    assert r[:3] == ["dubai_urban", "mecca_urban", "default"]
    assert r[3] is True
    assert "Dubai" in r[4]


# --- the gate ----------------------------------------------------------------------------


@needs_node
def test_only_local_osm_extrusions_are_themed():
    r = _node("""[urbanThemeApplies(urbanGeometryActive({active: false}, true)),
                   urbanThemeApplies(urbanGeometryActive({active: true}, true)),
                   urbanThemeApplies(urbanGeometryActive({active: true}, false)),
                   urbanThemeApplies(urbanGeometryActive(null, false)),
                   urbanThemeApplies("provider_3d_tiles"), urbanThemeApplies(null)]""")
    assert r == [True, False, False, False, False, False]


def test_the_provider_path_never_reads_the_theme():
    for word in ("THEME", "buildingColor", "roofShaders", "ROAD_STYLE", "urbanTheme"):
        assert word not in PHOTO, word
    code = re.sub(r"//.*", "", PAGE)
    # provider tiles are never restyled or reshaded by the page
    for word in ("Cesium3DTileStyle", "customShader", "CustomShader", "tileset.style"):
        assert word not in code, word


def test_the_theme_is_applied_inside_the_local_layer_only():
    code = re.sub(r"//.*", "", PAGE)
    outside = code.replace(re.sub(r"//.*", "", CITY_BLOCK), "").replace(re.sub(r"//.*", "", THEME_JS), "")
    for word in ("buildingColor(", "roofShaders(", "roadStyles(", "THEME."):
        assert word not in outside, f"{word} used outside the city layer"
    assert "buildingColor(THEME, rec)" in BUILD_CHUNK
    # the base-map toning is the local layer's too, and only on the OSM ground
    assert "if (URBAN && IMAGERY.mode === \"osm\") {\n    baseLayer.saturation = THEME.ground.saturation;" in CITY_BLOCK
    # the browser check reports whether the theme is on, from observed state
    assert "applied: urbanThemeApplies(urbanGeometryActive(prov, city.active && cc.shown > 0))" in PAGE


# --- the palette -------------------------------------------------------------------------


@needs_node
def test_colours_are_a_function_of_the_record_and_the_theatre_only(payload):
    recs = [r for ch in payload["chunks"] for r in ch["b"]]
    expr = f"""(() => {{
      const recs = {json.dumps(recs)};
      const t = urbanTheme("tehran_basin");
      const fwd = recs.map(r => buildingColor(t, r));
      const rev = recs.slice().reverse().map(r => buildingColor(t, r)).reverse();
      return {{ fwd, rev, hashes: recs.map(footprintHash) }};
    }})()"""
    a, b = _node(expr), _node(expr)
    assert a == b, "two runs of the same cache paint the same city"
    assert a["fwd"] == a["rev"], "chunk load order does not change a colour"
    assert all(0 <= h < 2 ** 32 for h in a["hashes"])
    assert len(set(a["hashes"])) == len(recs)
    for c in a["fwd"]:
        assert len(c) == 3 and all(isinstance(v, int) and 0 <= v <= 255 for v in c)


@needs_node
def test_the_palette_is_varied_and_uses_its_bands(payload):
    recs = [r for ch in payload["chunks"] for r in ch["b"]]
    out = _node(f"""(() => {{
      const recs = {json.dumps(recs)};
      const res = {{}};
      for (const k of Object.keys(URBAN_THEMES)) {{
        const t = URBAN_THEMES[k];
        res[k] = recs.map(r => buildingColor(t, r).join(","));
      }}
      return {{ res, rules: URBAN_THEME_RULES }};
    }})()""")
    towers = [i for i, r in enumerate(recs) if r[0] / 10 >= out["rules"]["towerM"]]
    assert towers and len(towers) < len(recs)
    for key, cols in out["res"].items():
        assert len(set(cols)) >= 0.5 * len(cols), f"{key}: neighbours must not all share a colour"
    # theatre-aware: one footprint, three cities, three looks
    t, d, m = out["res"]["tehran_basin"], out["res"]["dubai_urban"], out["res"]["mecca_urban"]
    assert sum(a != b and b != c and a != c for a, b, c in zip(t, d, m)) > 0.9 * len(recs)


@needs_node
def test_every_swatch_is_reachable_and_accents_are_a_minority():
    # synthetic low-rise records: the pick must reach every wall and accent
    # swatch, and accents stay near their documented share
    out = _node("""(() => {
      const t = URBAN_THEMES.dubai_urban, seen = new Map(); let acc = 0;
      for (let i = 0; i < 4000; i++) {
        const rec = [120, i * 37, i * 11 + 5, 3, 0, 0, 3, -3, 0];
        const h = footprintHash(rec);
        const low = (h & 0xff) < URBAN_THEME_RULES.accentShare * 256;
        const pool = low ? t.accents : t.walls;
        const sw = pool[(h >>> 8) % pool.length];
        seen.set(sw, (seen.get(sw) || 0) + 1); if (low) acc += 1;
      }
      return { seen: [...seen.keys()], acc, walls: t.walls, accents: t.accents };
    })()""")
    assert set(out["walls"]) | set(out["accents"]) == set(out["seen"])
    assert 0.06 < out["acc"] / 4000 < 0.14


def _palettes():
    block = THEME_JS[THEME_JS.index("const URBAN_THEMES"):THEME_JS.index("const URBAN_THEME_RULES")]
    out = {}
    for m in re.finditer(r"^  (\w+): \{(.*?)^  \},", block, re.M | re.S):
        body = m.group(2)
        pal = {k: re.findall(r"#[0-9a-f]{6}", re.search(rf"{k}: \[(.*?)\]", body).group(1))
               for k in ("walls", "towers", "accents", "roads")}
        pal["roof"] = re.search(r'roof: "(#[0-9a-f]{6})"', body).group(1)
        out[m.group(1)] = pal
    return out


@pytest.mark.parametrize("key", sorted(_palettes()))
def test_palettes_are_vibrant_but_restrained(key):
    p = _palettes()[key]
    walls = p["walls"]
    assert len(walls) >= 6 and len(set(walls)) == len(walls)
    old_s = _hsv(OLD_BUILDING)[1]
    sat = [_hsv(c)[1] for c in walls + p["towers"] + p["accents"]]
    assert sum(sat) / len(sat) >= 2 * old_s, "materially more colourful than the single warm grey"
    assert max(sat) <= 0.8, "restrained: no neon"
    for c in walls + p["towers"] + p["accents"]:
        assert 0.3 <= _hsv(c)[2] <= 0.98, f"{c}: neither black holes nor blown-out white"
    # a two-tone city at least (teal and gold, stone and olive), and a real
    # spread of light and dark within it, so neighbours separate
    hues = {round(_hsv(c)[0] * 12) % 12 for c in walls if _hsv(c)[1] > 0.12}
    assert len(hues) >= 2
    vals = [_hsv(c)[2] for c in walls]
    assert max(vals) - min(vals) >= 0.25
    # the roof tone is pale, so a tinted roof always reads lighter than its wall
    assert _hsv(p["roof"])[2] >= 0.9 and _hsv(p["roof"])[1] <= 0.15


@pytest.mark.parametrize("key", sorted(_palettes()))
def test_no_building_can_read_as_an_overlay(key):
    p = _palettes()[key]
    # An overlay is bright and saturated. A swatch may share a hue family with
    # one (terracotta is a red) but never its brightness and saturation both.
    for c in p["walls"] + p["towers"] + p["accents"]:
        h, s, v = _hsv(c)
        for name, o in OVERLAYS.items():
            oh = _hsv(o)[0]
            dh = min(abs(h - oh), 1 - abs(h - oh)) * 360
            assert dh > 20 or s < 0.6 or v < 0.85, f"{key} {c} too close to the {name} overlay"


# --- the roads ---------------------------------------------------------------------------


def test_the_page_styles_every_road_class_the_builder_emits():
    styles = PAGE[PAGE.index("const ROAD_STYLE = ["):PAGE.index("];", PAGE.index("const ROAD_STYLE = ["))]
    assert styles.count("{ w:") == len(ROAD_CLASSES) == 4
    for i in ROAD_CLASSES:
        assert f"{{ w: ROADS[{i}].w, c: rgba(ROADS[{i}].rgba) }}" in styles


@needs_node
def test_road_hierarchy_is_ordered_and_stronger_than_before():
    out = _node("Object.keys(URBAN_THEMES).map(k => [k, roadStyles(URBAN_THEMES[k])])")
    for key, styles in out:
        assert len(styles) == len(ROAD_CLASSES), key
        w = [s["w"] for s in styles]
        a = [s["rgba"][3] for s in styles]
        assert w == sorted(w, reverse=True) and len(set(w)) == len(w), f"{key}: widest first"
        assert all(n > o for n, o in zip(w, OLD_ROAD_WIDTHS)), f"{key}: wider than before"
        assert all(n >= o for n, o in zip(a, OLD_ROAD_ALPHA)), f"{key}: more opaque than before"
        assert a == sorted(a, reverse=True)
        # the major classes are coloured, not grey
        for s in styles[:2]:
            r, g, b = s["rgba"][:3]
            assert max(r, g, b) - min(r, g, b) >= 0.3, f"{key}: major roads carry colour"
        assert len({tuple(s["rgba"][:3]) for s in styles}) == len(styles)


@pytest.mark.parametrize("key", sorted(_palettes()))
def test_roads_stand_off_the_toned_ground(key):
    # the toned OSM ground sits around a mid grey; every road differs from it
    ground = (0.62, 0.62, 0.62)
    for c in _palettes()[key]["roads"]:
        rgb = _rgb(c)
        assert sum((x - y) ** 2 for x, y in zip(rgb, ground)) ** 0.5 >= 0.35, f"{key} {c}"


def test_roads_are_one_draped_primitive_per_chunk():
    # one GroundPolylinePrimitive per chunk: two on one road (a casing and a
    # fill) do not keep a stable draw order and strobe
    assert BUILD_CHUNK.count("new Cesium.GroundPolylinePrimitive(") == 1


# --- the roofs ---------------------------------------------------------------------------


@needs_node
def test_the_roof_patch_hits_the_stock_anchors():
    r = _node(f"roofShaders({json.dumps(STOCK_VS)}, {json.dumps(STOCK_FS)}, [243, 234, 219], 0.6, 0.84)")
    vs, fs = r["vs"], r["fs"]
    assert "out float v_up;" in vs and "in float v_up;" in fs
    assert "normalize(position3DHigh + position3DLow)" in vs
    assert "vec3(0.9529, 0.9176, 0.8588)" in fs and "0.600 * roof" in fs and "0.840" in fs
    # everything CesiumJS drew before is still drawn, in the same order
    assert vs.index("v_color = color;") < vs.index("v_up = ")
    assert "out_FragColor = czm_phong(" in fs and "material.alpha = color.a;" in fs
    assert fs.index("vec4 color = czm_gammaCorrect(roofed);") < fs.index("material.diffuse = color.rgb;")


@needs_node
def test_the_roof_patch_steps_aside_on_a_different_cesium():
    r = _node(f"""[roofShaders({json.dumps(STOCK_VS.replace("v_color = color;", "v_color = c;"))},
                               {json.dumps(STOCK_FS)}, [255, 255, 255], 0.5, 0.8),
                   roofShaders({json.dumps(STOCK_VS)}, "void main() {{}}", [255, 255, 255], 0.5, 0.8),
                   roofShaders(undefined, undefined, [0, 0, 0], 0.5, 0.8)]""")
    assert r == [None, None, None]
    # ... and the page then draws the stock shader, still per-building colour
    assert "vertexShaderSource: ROOF_SHADER ? ROOF_SHADER.vs : undefined" in CITY_BLOCK
    assert "fragmentShaderSource: ROOF_SHADER ? ROOF_SHADER.fs : undefined" in CITY_BLOCK


def test_the_stock_shaders_match_a_local_cesium_when_there_is_one():
    src = REPO / "node_modules" / "@cesium" / "engine" / "Source" / "Shaders" / "Appearances"
    vs, fs = src / "PerInstanceColorAppearanceVS.glsl", src / "PerInstanceColorAppearanceFS.glsl"
    if not (vs.exists() and fs.exists()):
        pytest.skip("no local @cesium/engine (node_modules is not tracked)")
    for anchor in ("void main()", "v_color = color;"):
        assert anchor in vs.read_text()
    for anchor in ("void main()", "vec4 color = czm_gammaCorrect(v_color);"):
        assert anchor in fs.read_text()


# --- the contract ------------------------------------------------------------------------


def test_buildings_stay_presentation_only_and_see_through_by_default():
    assert "occlusion: VISUAL.building_occlusion_default === true" in PAGE
    assert PAGE.count("urbanOverlaysOnTop = city.active && !city.occlusion;") == 2
    formula = PAGE[PAGE.index("const applyDepthTest = () => {"):PAGE.index("const xrayBtn")]
    assert "urbanOverlaysOnTop" in formula
    appearance = CITY_BLOCK[CITY_BLOCK.index("const buildingAppearance = () =>"):
                            CITY_BLOCK.index("function buildChunk(ci)")]
    assert "translucent: !city.occlusion" in appearance and "closed: true" in appearance
    # the instance colour carries the see-through alpha; toggling occlusion
    # swaps the appearance, never the geometry
    assert "URBAN_LIMITS.seeThroughAlpha) }," in BUILD_CHUNK
    assert "rec.b.appearance = buildingAppearance()" in CITY_BLOCK
    limits = PAGE[PAGE.index("const URBAN_LIMITS = {"):PAGE.index("};", PAGE.index("const URBAN_LIMITS = {"))]
    alpha = float(re.search(r"seeThroughAlpha: ([0-9.]+)", limits).group(1))
    assert 0.5 <= alpha <= 0.85, "still visibly see-through"
    assert "allowPicking: false, shadows: Cesium.ShadowMode.DISABLED" in BUILD_CHUNK


def test_roads_drape_on_the_terrain_only():
    assert BUILD_CHUNK.count("new Cesium.GroundPolylinePrimitive(") == \
        BUILD_CHUNK.count("classificationType: Cesium.ClassificationType.TERRAIN")
    assert "ClassificationType.BOTH" not in PAGE and "ClassificationType.CESIUM_3D_TILE" not in CITY_BLOCK


def test_buildings_still_stand_on_the_drawn_dem_and_load_in_chunks():
    assert "height: gmin - URBAN_LIMITS.footingM" in BUILD_CHUNK
    assert "extrudedHeight: gmax + rec[0] / 10" in BUILD_CHUNK
    assert BUILD_CHUNK.count("asynchronous: true") == 2
    assert "releaseGeometryInstances: true" in BUILD_CHUNK
    assert "rec.near = chunkDistanceM(" in CITY_BLOCK and "planChunks(" in CITY_BLOCK


def test_the_city_layer_writes_nothing_back():
    code = re.sub(r"//.*", "", CITY_BLOCK)
    for word in ("losLines", "threatEntities", "detectRings", "WebSocket", "postMessage",
                 "fetch(", "XMLHttpRequest", "ImageryProvider", ".png", ".jpg", "url(", "http"):
        assert word not in code, word
    assert 'getRoute("/urban")' in code


def test_the_page_still_credits_the_data_it_draws():
    assert "${VISUAL.base_attribution} ${URBAN_SUMMARY.attribution || \"\"} · cache ${city.data.cache_id}" in PAGE
    assert '"Visual only — urban presentation (local OpenStreetMap buildings)"' in PAGE
