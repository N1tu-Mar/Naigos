# DEVLOG

One entry per learning, architectural change or improvement. Newest at the bottom.

---

**Scoped the research agent to a fixed allowlist rather than open search.** Six sources, host-checked at fetch time, each carrying a license and a citation. The alternative — searching for whatever looks useful — would make the data layer unreproducible and would quietly invite exactly the scope drift the spec's guardrail exists to prevent.

**Chose Owens Valley as the primary theatre, but only after measuring it.** The AOI was picked for relief (Sierra crest to the west, White/Inyo range to the east, valley floor between) and then *tested*: 3396 m of relief and a monotonic masking gradient. A theatre is only worth anything here if terrain genuinely occludes, and that is a measurement, not a judgement call.

**Replaced a degenerate masking statistic with a real viewshed.** The first version asked "what fraction of the AOI is below the highest peak", which is trivially 100% and told us nothing. Replaced it with the actual per-ray LOS test from a summit sensor over 1500 random points. That is what produced the number the whole project rests on: 76% of the AOI is masked at 100 m above ground, 0% at 3000 m. The lesson is that a statistic which cannot come out badly is not evidence.

**Line-of-sight returns a signed clearance, not a boolean.** A hard 0/1 mask gives the policy no gradient — an aircraft 5 m below the sightline and one 500 m below look identical. Returning the minimum ray-above-terrain margin in metres makes the sign the constraint and the magnitude the learning signal.

**Cache keys embed an AOI bounds fingerprint.** Found this the expensive way: widening the AOI returned the previously cached DEM in 0 s and the extent statistics silently did not change. Without the fingerprint, every downstream LOS ray, spawn point and threat placement would have been computed against terrain that was not there.

**Cross-checked the DEM against published airfield elevations.** Two independent sources — 3DEP terrain and OurAirports field elevations — agree to within 3.6 m at every field in the theatre. This is the cheapest possible validation of the WGS84 → UTM reprojection that every LOS ray depends on, and a datum error would have shown up as tens of metres.

**The local atmosphere is not the textbook atmosphere.** Intending only to get air density for the airframe model, the pressure-level profile also yields the refractivity gradient, and therefore the effective-Earth factor the LOS model needs. Measured: −32.0 N-units/km, giving k = 1.256 against the standard 4/3 = 1.333. Dry high-desert air refracts less, so the radar horizon is genuinely shorter here than the default assumption. The LOS model takes k as a parameter and defaults to the measured value.

**Threat classes declare a design range; the range equation solves the power.** The first attempt hand-set transmit powers and apertures and produced a 1296 km detection range against a 1 m² target — unphysical, and beyond the horizon anyway. Inverting the problem means the only invented number per class is a deliberately round, role-based engagement distance, and everything else (RCS^(1/4) range scaling, frequency-dependent gaseous attenuation, the shape of the Pd ramp) follows from physics.

**Swerling 1 rather than a non-fluctuating detection model.** An aircraft's radar cross-section swings by an order of magnitude with small aspect changes. A non-fluctuating model turns detection on almost as a step (Pd 0.9 → 0.1 across a 1.2× range ratio); Swerling 1 spreads it across 2.2×. The blurrier edge is the physically right one *and* the one that gives the policy a continuous exposure gradient instead of a cliff.

**Albersheim's approximation now refuses to extrapolate.** Its argument goes negative outside roughly 0.1 ≤ Pd ≤ 0.9, where it was silently returning a domain error mid-inversion. It now raises with a message naming the validity band, and the inverter builds its grid only over the valid range. Returning a fabricated number outside a model's validity band is worse than failing.

**Validated the coordinated-turn relation against real traffic rather than asserting a g-limit.** 1459 consecutive ADS-B pairs of the same aircraft give a finite-difference turn rate that the instantaneous state vector does not carry. Inverting ω = g·tan(φ)/V puts implied bank angles at p95 9.1° and max 26.8° — exactly the range airliners fly. The relation the airframe model is built on is therefore confirmed against measurement; the manoeuvre limit itself remains an explicit design choice, flagged as such, because civil traffic never banks hard.

**The env reads component specs, never the cache.** `naigos/data/theatre.py` assembles everything the simulation is parameterized by from `components/*.json`. That indirection is what makes every number in the simulation traceable to a source and a sha256, and it keeps the training path free of any network or geospatial dependency.

**Edge sampling returned NaN on the southern and eastern grid boundaries.** `TerrainGrid.elevation` tested the *interpolable interior* (`r0 < rows-1`) rather than the raster footprint, so a point sitting exactly on the far edge came back as no-data and the ENU resampler then had to paper over a hole it should never have seen. Now "inside" means inside the bounds, and the outer half cell clamps to the edge cell.

**Two agents built overlapping layers; recorded rather than silently merged.** `naigos/data/dem.py` and `naigos/data/cache.py` arrived from the env side while the research layer was being built, duplicating the cited cache and the DEM path. The research-side resampler was therefore parked at `naigos/data/enu.py` instead of overwriting anything. The seams are listed in the handoff notes below; they are design decisions for a human, not something to reconcile unilaterally.

**Confirmed a vertical-mirror bug at the env/data seam.** `py3dep.get_dem` returns row 0 = north (y descending, EPSG:5070) — verified empirically, not assumed. `naigos/env/terrain.py::sample_height` indexes `hmap[y0, x0]` with `fy = y/cell`, so rows must increase with +y, i.e. row 0 must be south. `naigos/data/dem.py::load_dem` takes `raw.values` and resamples in index space with no flip and no reprojection out of Albers. The heightmap the env flies over is therefore mirrored north-south, and stretched anisotropically. This is precisely the bug class that never announces itself: the terrain still looks like terrain, LOS still returns plausible numbers, and the policy still learns something.
