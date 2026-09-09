# gaps.md — open issues in the visual-provider contract

Scope: the `--visual` work (`physics` / `photorealistic`) landed across commits
`548b6eb`, `822b60e`, `01e7f6a`, `86a92af`. Everything below is verified in the
working tree at `86a92af`, not inferred. 121 tests pass; these are the things the
tests do not cover, do not cover honestly, or actively get wrong.

Ordered by severity.

---

## 1. The Google Maps API key is served to the browser in every mode

**Severity: high. Credential exposure, and it contradicts a claim the code makes
about itself.**

`live.main` reads the key unconditionally and hands it to `make_handler`, which
substitutes it at `/*__GOOGLE_API_KEY__*/` on every page load:

```
naigos/demo/live.py:623   google_api_key = imagery_mod.resolve_google_api_key()
naigos/demo/live.py:631   make_handler(..., google_api_key=google_api_key)
naigos/demo/live.py:465   .replace("/*__GOOGLE_API_KEY__*/null", json.dumps(google_api_key))
```

Measured, by substituting a sentinel key into the real page template:

| `--visual` | ion token | route | key in served HTML | key actually needed |
| ---------- | --------- | ----- | ------------------ | ------------------- |
| `physics` | yes | – | **yes** | no |
| `photorealistic` | yes | `cesium_ion` | **yes** | no |
| `photorealistic` | no | `google_maps_api` | yes | yes |

Only the third row needs it. `cesium.html` guards its *use* correctly
(`if (VISUAL.tileset_route === "google_maps_api" && GOOGLE_API_KEY)`), but the
key is in the document source regardless, so anyone who can fetch `/` gets it —
including in the default mode, which never talks to Google at all.

This also falsifies the comment directly above the call site, which says nothing
token-shaped goes "into the page template" outside what the mode requires.

**Fix:** pass the key only when `visual.tileset_route == "google_maps_api"`, and
add the row above as a test. One line in `main`, one test.

---

## 2. The guard that keeps imagery out of `env`/`rl` does not know Google exists

**Severity: high. The invariant is load-bearing and the new provider is invisible
to it.**

`tests/test_imagery_layers.py:159` scans `naigos/env` and `naigos/rl` for words
that can only mean the viewer's skin:

```python
r"\bimagery\b|\bsentinel[-_ ]?2\b|\bbasemap\b|\bion_token\b|"
r"\bIonImageryProvider\b|\bsatellite\b|\borthophoto\b"
```

Nothing there matches `google`, `photorealistic`, `3d tiles`, `tileset`, or
`Cesium3DTileset`. A module under `naigos/env` could consume Google's tileset
today and the suite would stay green. The whole point of that test is that the
simulation packages cannot even *name* a visual provider.

**Fix:** extend the pattern. Watch for false positives — `tile` and `tiling`
appear in terrain code, so anchor on `3d[-_ ]?tiles`, `photorealistic`,
`tileset`, and `google` as whole words rather than on `tile`.

---

## 3. The viewer wiring has no tests at all

**Severity: high, and it is the part I did not write.**

A concurrent session (`session_01687rmcYzBKYYdaGf8xfebG`) added ~208 lines to
`naigos/demo/assets/cesium.html`: the tileset construction, a
`__GOOGLE_API_KEY__` substitution point, a visual-only banner, a runtime toggle
back to physics terrain, an LOD/request budget, and a tile-failure budget that
reverts the mode. `git show --stat` on that commit and on `cb30d25` shows the
only test files touched were `tests/test_visual_modes.py` (all 29 tests are
mine, all Python-side) and `tests/test_run_metadata.py` (unrelated).

So none of the following is pinned by anything:

- `const PHOTO = VISUAL.mode === "photorealistic"` — the only gate stopping
  physics mode from reaching `showPhotorealisticContext()`.
- `viewer.scene.globe.show = false` — the modelled surface is hidden in
  photorealistic mode. Nothing asserts it is *never* hidden in physics mode,
  which is exactly the E-9 failure shape in a new costume.
- `showPhysicsTerrain()` as the failure sink for a rejected key, an offline
  provider, a quota refusal, or `tileFailureBudget` exhaustion.
- `banner.hidden = false` — the attribution and non-evidence warning surface.
- The exaggeration reset (`EXAG_K !== 1` → `exagIdx = 0`), which exists because
  entity altitudes go through `exagZ()` and the tileset is unexaggerated.

`tests/test_imagery_layers.py` still passes only because the tileset is built
with `createGooglePhotorealistic3DTileset`, which does not contain the strings
`createWorldTerrain`, `CesiumTerrainProvider` or `fromIonAssetId` that the old
terrain guard looks for. That is luck, not coverage.

**Fix:** static assertions over the page text in the style of the existing
imagery tests — `globe.show = false` appears only inside the photorealistic
branch, `PHOTO` gates the entry point, `showPhysicsTerrain` is the catch target.
I have not reviewed that JS line by line; someone should before it is trusted.

---

## 4. `docs/DATA.md` does not list the new attribution-required source

**Severity: medium. Licence obligation, generated artifact, currently stale.**

`docs/DATA.md` is generated by `naigos-research` from `ALLOWLIST`, including an
"Attribution" table built from every source with `attribution_required=True`.
`google_photorealistic_3d_tiles` was added to the allowlist with that flag set,
but `DATA.md` in the tree still has no Google row — `grep -i google docs/DATA.md`
returns nothing.

The file only refreshes on a full `naigos-research` run, which needs the raw
cache populated, so it cannot be regenerated as part of this change.

**Fix:** re-run `uv run naigos-research --aoi tehran_basin` and commit the
regenerated `DATA.md`, or lift the attribution table generation into something
that runs without the cache.

---

## 5. Two entry points for the same question

**Severity: medium. Not a bug; a fork waiting to happen.**

`imagery.imagery_config(token, prefer=...)` survives as a shim over
`resolve_visual_config(...).to_page()`. It answers the narrow question ("which
skin, given this token") and `tests/test_imagery_layers.py` still calls it in
four places, so it cannot simply be deleted.

The risk is drift: a future rule added to `resolve_visual_config` that callers of
`imagery_config` silently miss. It also means "what is the public configuration
object" has two defensible answers, which is one too many.

**Fix:** either migrate the four call sites in `test_imagery_layers.py` and drop
the shim, or mark it explicitly deprecated in the docstring so nobody reaches for
it in new code.

---

## 6. A stale docstring now asserts something false

**Severity: low, but it is the kind that misleads a reader into a bad change.**

`tests/test_visual_modes.py:295`:

```python
def test_the_page_is_credited_for_what_it_actually_draws():
    """The viewer does not render the tileset yet, so it must not display a
    Google credit over pixels Google did not supply."""
```

The viewer does render the tileset now. The *assertion* is still right and still
worth keeping — `to_page()` credits the base layer because that is the provider
of the pixels under the cursor, and the tileset's credit belongs to the banner —
but the stated reason is obsolete, and reads as an invitation to delete the test.

I fixed the equivalent sentence in `imagery.py` in `01e7f6a` and missed this one.

**Fix:** one docstring.

---

## 7. Substitution-point coverage is incomplete

**Severity: low.**

`test_the_page_has_a_substitution_point_for_the_visual_contract` checks
`__VISUAL__`, `__IMAGERY__` and `__ION_TOKEN__`. It does not check
`__GOOGLE_API_KEY__`, which the other session added. If that placeholder is
renamed or dropped, `make_handler`'s `.replace()` becomes a silent no-op, the
page keeps `null`, and the `google_maps_api` route fails with a provider error
rather than a clear one.

The same test file's end-to-end substitution test (`:326`) has the same hole.

**Fix:** add the fourth placeholder to both.

---

## 8. Sentinel-2 is unreachable for the rest of a photorealistic session

**Severity: low. Design consequence, arguably correct, currently undocumented.**

`resolve_visual_config("photorealistic", ...)` forces `base_imagery="osm"` — the
right call, since Sentinel-2 tiles under an opaque tileset would be metered
against the ion quota and never seen (`validate_cli` refuses the explicit
combination for the same reason).

But `cesium.html`'s "physics terrain" button switches the *surface* back at
runtime while the base layer stays OSM. So a session started with
`--visual photorealistic`, with a valid ion token, can never show Sentinel-2 —
the user has to restart the server. Nothing tells them that.

**Fix:** either say so in the button's title text, or build both layers and
toggle visibility. The second costs an ion request the mode was designed to
avoid, so the first is probably right.

---

## Meta: how this work got committed

Worth recording because it affects how much of the above you should trust.

A second Claude session was editing the same files concurrently. It ran
`git add -A` and swept my staged contract into its own commit, `548b6eb`
("feat(bench): measure what a finer terrain grid costs and what it buys"). So:

- the visual-mode contract landed under an unrelated bench commit message;
- `cesium.html` carries viewer logic I did not write and have not line-by-line
  reviewed (gap 3);
- `live.py` was extended by that session after I wrote it — the
  `__GOOGLE_API_KEY__` path in gap 1 is theirs, not mine.

The tree is coherent and the suites pass. But "reviewed" and "passing" are not
the same claim, and only the second one is currently true of `cesium.html`.

---

## Suggested order

1, 2 first — both are security- or invariant-shaped and both are small. Then 3,
which is the largest and needs a read of the JS. 4 needs the data cache. 5–8 are
cleanup.
