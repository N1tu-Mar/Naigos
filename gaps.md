# gaps.md

Two concurrent workstreams, kept as two parts. Part 1 is the visual-provider
contract; Part 2 is the terrain-fidelity benchmark.

---

# Part 1 — open issues in the visual-provider contract

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

---
---

# Part 2 — open issues in the terrain-fidelity benchmark

Scope: `naigos/bench/terrain_resolution.py`, `scripts/bench_terrain.py`,
`tests/test_bench_terrain.py` and `docs/artifacts/terrain_bench.json`, landed
across `548b6eb`, `328dff0`, `12afd72`, `b7996e8`. Written by the second session
referred to above.

**On the collision.** Part 1's closing note is correct about the effect and
worth restating from the other side: commit `548b6eb` was made with
`git commit -F -` over an index that already held another session's staged
changes, so the visual-mode contract landed under a bench commit message. The
history was not rewritten to fix it, because a second session was committing at
the same time and rewriting shared history underneath it is worse than a
mislabelled commit. This document was then itself overwritten by that same
session and is restored here with both parts kept.

Open issues in the work in `naigos/bench/terrain_resolution.py`,
`scripts/bench_terrain.py` and `docs/artifacts/terrain_bench.json`. Written
while the benchmark was being built, so it records what the measurement does
*not* establish as well as what it does.

Each gap has an ID, why it matters, and what "done" looks like — the same shape
as `next-steps.md`, which is where these should migrate once triaged.

---

## What the benchmark now establishes

Measured on two AOIs (`owens_valley`, `tehran_basin`) × four cell sizes
(1500 / 500 / 250 / 100 m) × three ray-march lengths (`los_samples` 96 / 192 /
384), 64 worlds × 128 steps, best-of-9 timings, 2000 line-of-sight rays per row
against the native 30 m DEM. macOS arm64, 12 CPU cores, JAX 0.11.1, CPU backend.

1. **Grid resolution is very nearly free; the ray march is what costs.**
   At `los_samples=96`, throughput is flat within noise from 1500 m to 100 m
   cells (relative 1.00 / 0.98 / 1.00 / 0.93 on Owens, 1.00 / 1.03 / 0.99 / 1.03
   on Tehran) while the heightmap grows 230×. Doubling `los_samples` roughly
   halves throughput (1.00 / 0.54 / 0.30 and 1.00 / 0.52 / 0.25 at 500 m cells).
2. **The coarse grid, not the march, is the binding line-of-sight error.**
   At a fixed 96 samples, refining 1500 → 100 m cuts the false-visible rate from
   3.65% to 1.05% (Owens) and 5.35% to 1.40% (Tehran) at no throughput cost. At
   a fixed 1500 m grid, raising 96 → 384 samples moves the same rate only
   3.65% → 3.15% and 5.35% → 4.90%, for 3.4× the compute. Sampling a ridge that
   has already been averaged away does not recover it.
3. **Physics and presentation want different answers**, which is why the
   recommendation names two: 100 m at `los_samples=96` for physics, and 100 m
   served at `/terrain n=1024` for presentation. The presentation constraint is
   the endpoint's lat/lon resample, not the DEM: at the shipped `n=512`, a 100 m
   env grid renders with a p95 of 25.5 m against a 30 m AGL floor.

Nothing above changed a default. Every committed RL number was produced at
1500 m / 96 samples (500 m / 96 for the published re-measure and the live demo)
and still is.

---

## B-1 — The recommendation is measured on line-of-sight geometry, not on policy outcomes — BLOCKING for adoption

The benchmark grades a resolution by how often it disagrees with the source DEM
about whether a ray is cut. It does **not** measure what changes downstream:
survival, objective rate, mean detection probability, the altitude the policy
settles at.

**Why it matters.** A 1.4%-vs-3.65% false-visible rate is a statement about the
sensor model, not about the task. The repo has already been bitten by the
converse — `docs/DEVLOG.md` records that an earlier resolution conclusion was
drawn from high-altitude rollouts where grid refinement genuinely does nothing,
and was wrong for the low-altitude case. A geometry benchmark can make the same
mistake in the other direction.

**Done looks like.** The committed checkpoint evaluated on the demo seeds at
1500 / 500 / 250 / 100 m, reported as a table beside the geometry numbers. If
survival and detection move by less than seed noise, the honest conclusion is
that the resolution choice is not load-bearing for the published claim, and the
benchmark should say so.

## B-2 — `false_visible_rate` thresholds are asserted, not derived

`ResolutionPolicy.max_false_visible_rate = 0.02` and
`max_soft_visibility_mae = 0.05` are numbers I chose. They are stated in one
place and the recommendation is a pure function of them, which is the most that
can be said for them.

**Why it matters.** 250 m at 96 samples fails on Tehran by 0.0210 against 0.0200.
One threshold digit moves the recommendation from 100 m to 250 m. A reader is
entitled to know the answer is that sensitive.

**Done looks like.** Either B-1 supplies an outcome-anchored budget (what
false-visible rate perturbs survival by less than seed noise), or the report
publishes the recommendation across a threshold sweep so the sensitivity is
visible rather than implicit.

## B-3 — The `ray_step_cells` heuristic predicts the opposite of what was measured

The advisory diagnostic says that at 100 m cells and 96 samples the march steps
13.5–17.4 cells per sample along the theatre diagonal, i.e. most of the grid is
never looked at. The end-to-end measurement says that same configuration is the
most accurate one at full throughput.

**Why it matters.** Both are reported and they disagree, so at least one is
being misread. The likely explanation is that a bilinear 100 m surface preserves
ridge *height* even when sampled sparsely, whereas a 1500 m surface has already
averaged the ridge into the valley and no sampling density recovers it — but
that is a hypothesis, not a measurement, and it is currently written into the
module as a comment rather than tested.

**Done looks like.** A ridge-crossing test: synthetic terrain with a single
known ridge of known width, sampled at each cell size, showing what fraction of
the ridge height survives resampling. That separates "the march missed it" from
"the grid never had it".

## B-4 — Only the shipped endpoint resample is measured, not the fix already proposed

Presentation is graded on `build_terrain_grid`'s lat/lon resample at
n ∈ {256, 512, 1024}. `next-steps.md` E-12 already proposes replacing it with
the ENU grid plus a small lon/lat→ENU warp table, which would be accurate to
under a metre and *shrink* the payload from 512 KB to 65 KB.

**Why it matters.** The presentation recommendation (100 m at n=1024, a 2 MB
payload) is the best answer available inside the current design. It is a worse
answer than the design change that is already written down. Recommending 2 MB
without saying that would be misleading.

**Done looks like.** E-12 implemented, then re-benchmarked; the expected result
is that the presentation cell size stops being constrained by the endpoint at
all.

## B-5 — CPU only, one machine, and it was a shared one

Every number is from a single macOS arm64 laptop on the CPU backend, and part of
the sweep ran while another job was training on the same machine. That is why
`throughput_stats` reports best-of-N with the spread beside it rather than a
median: the same configuration came back at 186k, 153k and 139k agent-steps/s
across three medians-based runs, which is enough spread to invert the decision.

**Why it matters.** The cell-size-is-free finding is a statement about this CPU's
cache behaviour. On a GPU, a 5.4 MB heightmap and a 14 KB one have very
different residency characteristics, and the conclusion could reverse.

**Done looks like.** The same sweep run on the Modal GPU worker (`C-1`), with
`device` memory stats populated — they are `null` here because the CPU backend
implements no allocator — and the two conclusions compared.

## B-6 — Line-of-sight rays are drawn uniformly, not from the flown distribution

`sample_ray_endpoints` draws both ends uniformly over the theatre, at a fixed
10 m AGL emitter and 150 m AGL target. Real rays run from threat sites, which
are placed by spawn geometry, to aircraft, which fly where the policy takes them
— and the trained policy climbs to ~5400 m (`next-steps.md` V-1), where masking
barely operates.

**Why it matters.** The measured false-visible rates describe a low-altitude
ray population. That is the right population for the mechanic the project is
about, but it is not the population the current policy generates, so the numbers
overstate how much this resolution choice affects *this* checkpoint.

**Done looks like.** A second ray set drawn from a logged rollout's actual
(threat, aircraft) pairs, reported alongside the uniform one. The gap between
the two is itself the interesting number.

## B-7 — `TerrainConfig.cell` and the demo's `--cell-m` are still one knob

`naigos.demo.live` passes a single `--cell-m` to `env_from_theatre`, so raising
presentation fidelity necessarily rebuilds the simulation grid, and vice versa.
The benchmark now shows the two want different values.

**Why it matters.** As long as they are one number, "make the globe look better"
silently changes the surface line-of-sight is computed against — which is the
class of defect E-9 already cost this repo once.

**Done looks like.** A separate presentation cell size threaded to
`build_terrain_grid` only, with a test asserting the physics grid is unaffected.
Deliberately **not** done here: the task was to improve the decision, not to
change the configuration behind the published results.

## B-8 — No CI, so the benchmark can rot silently

`scripts/bench_terrain.py` is run by hand. `tests/test_bench_terrain.py` covers
the pure functions, so the arithmetic cannot rot, but nothing re-runs the sweep
or checks that `docs/artifacts/terrain_bench.json` still matches the code that
produced it.

**Done looks like.** `--quick` wired into the CI workflow that `next-steps.md`
G-2 already asks for, asserting only that the sweep completes and the schema
validates — not the timings, which are not reproducible on a shared runner.
