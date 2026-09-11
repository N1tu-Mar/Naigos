# Urban local theme

`--visual urban-presentation` draws a city in one of two ways. Google
Photorealistic 3D Tiles come with the provider's own materials, and Naigos
does not restyle them. Local OpenStreetMap extrusions are Naigos's own
geometry, and this theme applies only to them. It colours the buildings, gives
roofs a separate tone from walls, and sets a width hierarchy for roads, so that
Tehran, Dubai and Mecca each have a distinct, readable look.

Everything here is presentation. The simulation never sees a building, and the
theme does not change that.

## When it applies

The theme applies only while the HUD reads **local OSM buildings**. That HUD
text comes from `urbanGeometryActive()`, and `urbanThemeApplies()` accepts only
its `local_osm_extrusions` result. The browser check
`NAIGOS_VIEW.urban().theme` reports:

```json
{"key": "dubai_urban", "name": "Dubai: teal and gold", "roof_shader": true, "applied": true}
```

`applied` is computed from observed renderer state, the same way the HUD is.
When provider tiles are active (**provider buildings active**), `applied` is
false. The provider block (`showPhotorealisticContext`) never reads the theme,
and the page contains no `Cesium3DTileStyle` or custom shader for tiles.

## Building colour

`URBAN_THEMES` in `naigos/demo/assets/cesium.html` holds one palette per city
theatre, plus a neutral `default` for any other theatre:

| theatre        | walls                                      | towers (≥ 40 m)               |
|----------------|--------------------------------------------|-------------------------------|
| `tehran_basin` | terracotta, brick, apricot, stone, slate, grey-teal | steel blue, glass teal, silver |
| `dubai_urban`  | sand, gold, champagne, teal, turquoise, white stone | glass teal, aqua, gold, silver, champagne |
| `mecca_urban`  | sandstone, ochre, olive, sage, clay, cream  | pale sandstone, olive grey, bronze |

About 10% of lower buildings take an accent swatch instead of a wall swatch.

Each building's colour comes from `footprintHash(rec)`: FNV-1a over the
footprint record's integers (height in decimetres and quantised vertices),
followed by murmur3's finaliser. The hash picks the swatch and adds ±8%
lightness, so neighbouring buildings that share a swatch still look separate.
Only integer arithmetic is used, with no clock and no randomness, so:

- the same cache and theatre always produce the same colours;
- colours do not depend on the order in which chunks load;
- a rebuilt cache with the same footprints keeps its colours.

Palettes are chosen by the payload's `aoi`, falling back to `scene.theatre`
and then to `default`. The only input is the footprint. Height matters only
because towers use the glassier set, and height is already visible in the
extrusion. The payload has no use, owner or category tag, so the colours cannot
carry any real-world operational meaning. `tests/test_urban_theme_page.py`
checks that no swatch is both as bright and as saturated as the page's red
(threat) or green (detection) overlay colours.

## Roofs

Buildings use `PerInstanceColorAppearance` with CesiumJS's stock shaders.
`roofShaders()` patches those shaders at two text anchors each. In the vertex
shader it compares the face normal with the geocentric up direction. In the
fragment shader it blends upward-facing faces (roofs) toward the theme's pale
roof tone and shades walls down to 84%. The result is a light roof over a
darker wall of the same hue.

If CesiumJS changes its shader text and an anchor is missing, `roofShaders()`
returns `null` and the page falls back to the stock shaders. Per-building
colour still works in that case; only the roof tint is lost. `roof_shader` in
`NAIGOS_VIEW.urban().theme` shows which of the two paths is running. CesiumJS
is pinned to 1.145 (see `tests/test_cesium_version.py`), and the test file
includes that version's shader source.

## Roads

| class | builder source                   | width (px) | alpha | was            |
|-------|----------------------------------|-----------:|------:|----------------|
| 0     | motorway, trunk                  | 6.0        | 0.96  | 4.0 px @ 0.85  |
| 1     | primary, secondary               | 4.4        | 0.93  | 2.6 px @ 0.75  |
| 2     | tertiary                         | 3.0        | 0.88  | 1.6 px @ 0.60  |
| 3     | residential (opt-in per city)    | 1.9        | 0.78  | 1.1 px @ 0.50  |

Colours go from warm orange (class 0) through amber and cream to a pale warm
grey, with each theatre using its own tints. Roads have no casing. A dark
casing drawn as a second `GroundPolylinePrimitive` under each fill does not
keep a stable draw order in CesiumJS's ground-polyline volumes, and it showed
through the fill as black-and-white dashes. Contrast comes instead from colour,
width, and a darker toned ground: under the city the OSM base drops to
brightness 0.62–0.64 and saturation 0.26–0.30, down from 0.82 and 0.35.

## What is unchanged

- **Building occlusion off.** This is still the default, and it means the same
  thing: buildings are translucent (`translucent: !city.occlusion`) and every
  marker is drawn on top (`urbanOverlaysOnTop`, `applyDepthTest`). The
  see-through alpha goes from 0.6 to 0.8 so the palette shows up as colour.
  Buildings stay visibly translucent, and markers-on-top does not depend on the
  alpha. Turning occlusion on switches the appearance to an opaque one; the
  geometry is not rebuilt.
- **Terrain-only roads.** Each chunk has one `GroundPolylinePrimitive`, with
  `ClassificationType.TERRAIN`.
- **Loading.** Chunked asynchronous primitives, the initial cap, proximity
  loading, distance culling and sliced main-thread work all work as before.
  Buildings still stand on `sampleGrid()` heights.
- City toggles, protected-zone cut-outs (applied by the builder), attribution
  and the `cache …` credit are unchanged.
- No network. The theme adds no request, raster asset or tile style. All
  colours are inline constants.

## Verifying

```bash
uv run pytest tests/test_urban_layer.py tests/test_city_presentation.py \
  tests/test_urban_theme_page.py -q
```

To inspect a city with no credentials, so it can only use the local path:

```bash
NAIGOS_CESIUM_ION_TOKEN= CESIUM_ION_TOKEN= NAIGOS_GOOGLE_MAPS_API_KEY= GOOGLE_MAPS_API_KEY= \
uv run python -m naigos.demo.live --aoi dubai_urban --checkpoint checkpoints/theatre_1000.pkl \
  --visual urban-presentation --camera urban-overview --open
```

The HUD should read **local OSM buildings · N of M drawn · K road pieces**, and
`NAIGOS_VIEW.urban().theme.applied` should be `true`.
