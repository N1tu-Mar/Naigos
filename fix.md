# fix.md — Part 1 of gaps.md, closed

Companion to [gaps.md](gaps.md). Fixes 1 and 2 below are the two
high-severity items, in commits `e4a60b7` and `f236970`. Fixes 3–8 close the
rest of the visual-provider list; every item in Part 1 is now shut.

Part 2 — the terrain-fidelity benchmark, `B-1` through `B-8` — is untouched.

---

## Fix 1 — the Google Maps key now reaches the page only on the route that uses it

**Was:** `live.main` read the key unconditionally and `make_handler` substituted
it at `/*__GOOGLE_API_KEY__*/` on every page load, in every mode. `/` is served
to whoever can reach the port, so the key was handed out in two cases out of
three that cannot use it — including `--visual physics`, the default, where
CesiumJS never contacts Google at all.

**Now:** gated on the resolved route.

```python
# naigos/demo/live.py, in render_page()
page_google_key = (
    google_api_key if visual.tileset_route == "google_maps_api" else None
)
```

Measured against the real page template, before and after:

| `--visual` | ion token | route | key in HTML *before* | *after* | needed |
| ---------- | --------- | ----- | -------------------- | ------- | ------ |
| `physics` | yes | – | yes | **no** | no |
| `photorealistic` | yes | `cesium_ion` | yes | **no** | no |
| `photorealistic` | no | `google_maps_api` | yes | yes | yes |

### The structural half

The substitution moved out of `make_handler`'s closure into a module-level
`render_page(visual, ion_token, google_api_key)`. Two reasons, and the second is
the real one:

- the gate now lives in one place, so every caller inherits it rather than
  having to remember it at the call site;
- it is the only function in the server that touches secrets, and it is now
  directly testable — no env, no checkpoint, no JIT standing between the
  assertion and the string it is asserting about.

`make_handler` keeps its signature and calls it.

### Tests added

`tests/test_visual_modes.py`, six new:

- three parametrized cases pinning the table above, each also asserting the
  placeholder is consumed either way — a surviving `/*__GOOGLE_API_KEY__*/`
  would mean the `.replace()` had silently become a no-op;
- the ion token appears in the rendered page exactly once, and not at all when
  absent;
- neither credential survives into the config blobs that the page, `/scene` and
  stdout all share;
- `__GOOGLE_API_KEY__` added to the substitution-point test (gap 7, closed as a
  side effect).

---

## Fix 2 — the simulation guard knows about the photorealistic provider, and about identifiers

`tests/test_imagery_layers.py` scans `naigos/env` and `naigos/rl` for words that
can only mean a viewer visual provider. Nothing may name one; if it cannot name
one, it cannot be reading one.

### The gap as reported

The pattern matched `imagery`, `sentinel-2`, `basemap`, `ion_token`,
`IonImageryProvider`, `satellite`, `orthophoto` — and nothing whatsoever from the
family the `--visual` work had just introduced. A module under `naigos/env` could
have consumed Google's 3D Tiles with the suite green.

Added: `google`, `photo-realistic`, `tileset(s)`, `3d tiles`, `Cesium3DTileset`.

Deliberately still excluded, and now asserted rather than left to a comment:
bare `sentinel` (a sentinel index in the spatial hash), bare `Cesium` (a comment
about what the georef is *for*), bare `ion` (`runmeta.py` explaining why
`run.json` never captures `os.environ` — a docstring about *avoiding*
credentials), and bare `tile` (terrain tiling code).

### The second gap, which the teeth test found

Writing the "would this catch a real violation" test first turned up a hole
older than this change. `_` is a word character in Python's regex, so:

```
\bimagery\b     does NOT match  fetch_imagery(bbox)
\bsatellite\b   does NOT match  satellite_patch
\bbasemap\b     does NOT match  basemap_cache
```

Verified against the pre-existing pattern — all three returned `False`. An
identifier is exactly the shape a real violation would take, so the guard was
blind to its most likely form for as long as it has existed.

The boundary is now alphanumeric-edged, so `_` separates:

```python
_EDGE_L, _EDGE_R = r"(?<![A-Za-z0-9])", r"(?![A-Za-z0-9])"
```

All three now match, and the four exclusions above still do not.

### Tests added

The pattern moved to a module-level `FORBIDDEN_IN_SIMULATION` shared by three
tests, so a typo in the alternation cannot disable the scan while leaving its
own teeth test green:

- `test_the_guard_would_catch_a_real_violation` — nine lines that must trip it,
  including the three identifier forms above and a plausible
  `Cesium3DTileset.fromIonAssetId(2275207)`;
- `test_the_guard_does_not_fire_on_the_simulations_own_vocabulary` — four lines
  from the real codebase that must not;
- the scan itself, unchanged in shape.

---

## Fix 3 — the viewer wiring is pinned (`f6b2d5b`)

Closed by `tests/test_visual_renderer.py`, 33 static assertions over the page
text, no credential, no network and no browser. The claims gap 3 listed as
pinned by nothing are now each an assertion: `PHOTO` gates the entry point;
`globe.show` is written in exactly two places, one per mode, so no path can put
two surfaces up at once; `showPhysicsTerrain` is the catch target, the
`tileFailed` budget's sink, and where the request budget is handed back; the
banner has one way in and one out; the exaggeration reset is asserted because
entity altitudes go through `exagZ()` and the tileset does not.

Two things the file adds beyond the gap's list. `test_the_mode_switch_does_not_touch_the_overlays`
scans the comment-stripped block for each of eleven overlay names, so the mode
machinery cannot start reaching into world-space entities. And
`test_no_google_api_key_is_committed_to_the_repository` scans every tracked file
for the `AIza`+35 shape, the same posture the ion token already had.

## Fix 4 — `docs/DATA.md` credits the photorealistic provider

The doc is generated by `naigos-research`, which needs the raw cache, so between
an `ALLOWLIST` change and the next full run the tree stated obligations that
were out of date — which is tolerable for a measurement report and not for a
licence.

The two sections that depend only on the allowlist are now
`naigos.research.run.source_sections()`, called by `write_data_doc` and by a new
`naigos-research --refresh-docs` that rewrites that span in place with no cache,
no network and no AOI. The attribution prose gained a paragraph naming Google
Photorealistic 3D Tiles as presentation-only, since crediting the provider
without saying the surface is not the DEM would read as a provenance row for
terrain the detection model consumes.

`tests/test_data_doc.py` — 16 tests — asserts every allowlisted source has a
Sources row, every `attribution_required` source has a citation row, and that
the committed span still equals what the allowlist generates today. The last
one is what catches a reworded citation, and it fails with the command to run.

## Fix 5 — one entry point, not two

`imagery.imagery_config(token, prefer=...)` is deleted. The five call sites in
`tests/test_imagery_layers.py` now call
`resolve_visual_config(ion_token=..., imagery=...).to_page()`, which is what the
shim did anyway; `VisualConfigError` subclasses `ValueError`, so the refusal
test needed no change.

`test_there_is_exactly_one_public_way_to_ask_what_the_globe_is_showing` keeps it
gone. The risk was never the shim's behaviour, it was drift: a rule added to the
resolver that its callers silently miss.

## Fix 6 — the stale docstring

`test_the_page_is_credited_for_what_it_actually_draws` said the viewer does not
render the tileset yet. It does. The assertion was always right for a different
reason, now written down: the blob is the base layer alone, so a Google credit
in it would sit over OpenStreetMap pixels showing through where the tileset has
no coverage.

## Fix 7 — substitution-point coverage

Closed as a side effect of fix 1.

## Fix 8 — the page says what the mode costs for the session

`--visual photorealistic` forces `base_imagery="osm"`, and the runtime toggle
returns the surface rather than the skin, so a session started in that mode can
never show Sentinel-2. Building both layers and switching visibility would spend
exactly the ion requests the mode exists to avoid, so the page tells the truth
instead: `imgBtn.title` says Sentinel-2 needs a restart without
`--visual photorealistic`.

Gated on `VISUAL.ion_token_present`, because without a token OSM is what physics
mode would have drawn anyway and there is nothing lost to report. Two tests: one
for the note, one asserting the mode block never constructs an imagery provider
or touches `baseLayer`, so the alternative fix cannot creep in unannounced.

---

## What this did not touch

Part 2 of [gaps.md](gaps.md) — `B-1` through `B-8`, the terrain-fidelity
benchmark. `B-1` is the blocking one: the resolution recommendation is graded on
line-of-sight geometry and never on policy outcomes.

Two things found while closing Part 1 and deliberately left alone, both in
`naigos/rl/runmeta.py`, which a concurrent session was editing at the time:

- `tests/test_imagery_layers.py::test_no_satellite_pixel_can_reach_the_policy[rl]`
  fails on its secret-redaction pattern, which names `ION_TOKEN` in order to
  scrub it. The guard is doing its job on a line that is arguably the one
  legitimate reason for `naigos/rl` to name the token; whoever owns that file
  should decide whether the scrubber belongs there or whether the guard needs an
  exemption with a reason attached.
- The reviewed claim from gap 3 now holds for `cesium.html`'s visual-mode block,
  which is pinned test by test. The rest of that file — the replay path, the
  static export — is passing, not reviewed.
