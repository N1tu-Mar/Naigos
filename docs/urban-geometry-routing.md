# Urban geometry routing

`--visual urban-presentation` draws a city for context. Its buildings come from
one of two sources:

| source | `geometry_source` | what it is |
|---|---|---|
| local | `local_osm_extrusions` | OpenStreetMap footprints and major roads from the local visual cache (`naigos.demo.urban`), extruded over the simulation's own DEM. Naigos owns and styles this layer. |
| provider | `provider_3d_tiles` | Google Photorealistic 3D Tiles, streamed by CesiumJS via Cesium ion or the Google Maps Tiles API. The provider owns the geometry and the texture. |

Both are presentation only (`evidence_grade: false`). Neither is used by terrain
LOS, detection, RL or any simulation result. Render-only building occlusion
starts OFF in both.

## Before this change

`naigos/demo/imagery.py:_resolve_urban` picked the source from whatever
credentials were in the environment:

```python
route = "cesium_ion" if has_ion else ("google_maps_api" if has_google else None)
if route is not None:
    return VisualConfig(..., tileset="google_photorealistic",
                        geometry_source=GEOMETRY_PROVIDER, ...)
```

`naigos/demo/assets/cesium.html` treats any config with a tileset as the
provider page:

```js
const PHOTO = VISUAL.mode === "photorealistic" || (URBAN && !!VISUAL.tileset);
...
if (PHOTO) showPhotorealisticContext();   // hides the globe
```

and `citySurfaceChanged()` sets `city.active = !!city.data && onDem`, so the
local city layer is hidden whenever the provider surface is up.

The result: anyone with `NAIGOS_CESIUM_ION_TOKEN` exported, which is the normal
setup for Sentinel-2 in physics mode, got Google's buildings under
urban-presentation, and the Naigos-styled local layer only showed up as a
runtime fallback. There was no way to ask for the local layer while a
credential was set.

## The contract

One knob, `--urban-geometry`, sets the policy. Credentials never do.

```
--urban-geometry local      # default
--urban-geometry provider   # explicit opt-in to Google Photorealistic 3D Tiles
--urban-geometry auto       # provider when a credential exists, else local (the old behaviour)
```

`imagery.resolve_visual_config(..., urban_geometry=None)` takes the same values;
`None` means the default, `local`.

### Routing table

| policy | credential + cache | credential, no cache | no credential + cache | nothing |
|---|---|---|---|---|
| `local` (default) | **local** | unavailable | local | unavailable |
| `provider` | provider | provider | local (fallback) | unavailable |
| `auto` | provider | provider | local | unavailable |

"unavailable" means `geometry_source: null` and the labelled
`urban data unavailable` state, with a `fallback_reason` that names the fix.

### What each outcome reports

These fields come from `VisualConfig.as_dict()`, which is embedded in the page
as `VISUAL` and served in `/scene` as `visual`. `--smoke-render` prints the same
values.

| outcome | `urban_geometry` | `requested_geometry_source` | `geometry_source` | `tileset` | `provider_state` | `fallback_reason` |
|---|---|---|---|---|---|---|
| local, cache present | `local` | `local_osm_extrusions` | `local_osm_extrusions` | `null` | `not_requested` | `null` |
| local, no cache | `local` | `local_osm_extrusions` | `null` | `null` | `not_requested` | `urban data unavailable: no local urban cache for this AOI. Build it once with: uv run python -m naigos.demo.urban --aoi <aoi>` (plus "or opt in … with --urban-geometry provider" when a credential is set) |
| provider, credential | `provider` | `provider_3d_tiles` | `provider_3d_tiles` | `google_photorealistic` | `awaiting_browser` | `null` |
| provider, no credential, cache | `provider` | `provider_3d_tiles` | `local_osm_extrusions` | `null` | `unavailable_no_credentials` | `--urban-geometry provider needs a Cesium ion token (...) or a Google Maps Tiles API key (...); neither is set, so the local cached OpenStreetMap building layer is drawn ...` |
| provider, nothing | `provider` | `provider_3d_tiles` | `null` | `null` | `unavailable_no_credentials` | `urban data unavailable: --urban-geometry provider needs ...; neither is set, and there is no local urban cache. Build it once with: ...` |
| auto, credential | `auto` | `provider_3d_tiles` | `provider_3d_tiles` | `google_photorealistic` | `awaiting_browser` | `null` |
| auto, no credential, cache | `auto` | `local_osm_extrusions` | `local_osm_extrusions` | `null` | `unavailable_no_credentials` | `provider buildings need ...; neither is set, so the local cached OpenStreetMap building layer is drawn ...` |

When `requested_geometry_source` and `geometry_source` differ, a fallback
happened, and `fallback_reason` explains it. The server never reports the
provider as `active`: only the browser can see a tile on screen.

### What the browser reports

The page is unchanged. It already decides from what the renderer actually drew,
not from the config:

- The HUD line (`#urbanstatus`) reads **`local OSM buildings · N of M drawn · K
  road pieces`** once local chunks are shown. Under the default policy
  `providerProbe.requested` is false, so there is no provider suffix.
- `window.NAIGOS_VIEW.urban()` returns `configured_geometry:
  "local_osm_extrusions"`, and `geometry_active: "local_osm_extrusions"` only
  once buildings are shown (`urbanGeometryActive(prov, city.active && shown >
  0)`).
- With no cache, the HUD reads **`URBAN DATA UNAVAILABLE — no buildings are
  drawn.`** followed by the `/scene` urban reason or `VISUAL.fallback_reason`.
  Both include the `naigos.demo.urban` build command. `geometry_active` is
  `null`.
- Under `--urban-geometry provider`, the HUD says `provider buildings active`
  only after a tile has loaded and become visible, as before.

The routing change only has to make `VISUAL.tileset` null whenever the local
layer is the chosen source. With no tileset the page never enters `PHOTO`, the
globe stays on the simulation DEM, and `citySurfaceChanged()` activates the
local layer.

### Credentials

- `VisualConfig` still holds only the booleans `ion_token_present` and
  `google_api_key_present`. Tests serialise `as_dict()`, `to_page()`, `repr()`
  and `describe()` under every policy and credential combination and check that
  no secret appears.
- `live.render_page` now keeps the ion token out of an urban-presentation page
  that is not on the `cesium_ion` route. That page's base layer is keyless OSM
  and its buildings are local or come straight from Google, so nothing on it
  talks to Cesium ion. The Google key already reached the page only on the
  `google_maps_api` route. Physics and photorealistic pages are unchanged.
- Static exports (`naigos.demo.viewer`) pass no credentials and resolve to
  local, as they did before.

## Commands

```bash
# build the local layer once (network, allowlisted Overpass source)
uv run python -m naigos.demo.urban --aoi tehran_basin

# default: local OSM buildings, even with NAIGOS_CESIUM_ION_TOKEN exported
uv run python -m naigos.demo.live --aoi tehran_basin --visual urban-presentation

# opt in to Google Photorealistic 3D Tiles (needs a credential)
uv run python -m naigos.demo.live --aoi tehran_basin --visual urban-presentation \
    --urban-geometry provider

# see the resolved routing without a browser
uv run python -m naigos.demo.live --aoi tehran_basin --visual urban-presentation --smoke-render
```

At startup, under the default policy with a credential set, the terminal says:

```
visuals: urban-presentation -- local cached OpenStreetMap buildings and roads over the
simulation DEM (a provider credential is set but not used: Google Photorealistic 3D Tiles
are opt-in with --urban-geometry provider). PRESENTATION MODE: ...
```

`--urban-geometry` with any other `--visual` is refused at startup. A flag that
silently does nothing reads as a flag that worked.

## Scope

This change is visual only. It does not touch simulation terrain, LOS,
detection, RL, protected-zone exclusions, the data-fetching rules or
building-occlusion semantics.

## Also on this branch: achieved live rate

The HUD's `live · 10× real time` is the `--speed` request. Each live frame now
also carries the measured rate:

```json
"live_rate": {"label": "achieved", "achieved_sim_s_per_wall_s": 9.87,
              "requested_sim_s_per_wall_s": 10.0, "fraction_of_requested": 0.987,
              "window_wall_s": 5.01}
```

`live.AchievedRate` computes it as sim time advanced divided by wall time
elapsed, over a rolling window of at least 5 s of wall clock. It is fed after
each step and never read back into the step, `dt` or the sleep schedule. Replay
payloads carry no stream rate. A replay's playback speed belongs to the page's
clock widget.

To show it in the HUD, `cesium.html` needs one line, which is outside this
branch's files. In the stream handler, after a frame is parsed:

```js
if (!REPLAY && f.live_rate) document.getElementById("rate").textContent =
  `live · ${scene.speed}× requested · ${f.live_rate.achieved_sim_s_per_wall_s ?? "…"}× achieved`
  + ` · ${scene.n_blue} aircraft · CBF ${scene.cbf ? "on" : "off"}`;
```
