# fix.md — gaps 1 and 2 closed

Companion to [gaps.md](gaps.md). Covers the two high-severity items from that
list. Commits `e4a60b7` and `f236970`. 151 tests pass across the demo,
provenance, env and invariant suites.

Gaps 3–8 are untouched and still open.

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

## What this did not touch

From [gaps.md](gaps.md), still open:

| # | | severity |
| - | - | -------- |
| 3 | `cesium.html` viewer wiring has no tests and is unreviewed | high |
| 4 | `docs/DATA.md` missing the Google attribution row (needs a research run) | medium |
| 5 | `imagery_config` shim duplicates `resolve_visual_config` | medium |
| 6 | stale docstring on `test_the_page_is_credited_for_what_it_actually_draws` | low |
| 8 | Sentinel-2 unreachable for the rest of a photorealistic session | low |

Gap 7 (substitution-point coverage) closed as part of fix 1.

Gap 3 remains the one worth attention: `createGooglePhotorealistic3DTileset`,
the `globe.show = false` switch, the failure sinks and the banner are all still
pinned by nothing, and that file was written by a concurrent session rather than
reviewed here.
