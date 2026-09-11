"""Offline fixtures for the pipeline tests: a snapshot tree with no network and no cache.

Built with the real `cache.record` and `spec.write_component`, pointed at a
temporary root, so the tree has exactly the shape the research agent writes --
only the bytes are small and fake.
"""

from __future__ import annotations

from pathlib import Path

from naigos.research import cache, roots, spec
from naigos.research.allowlist import GUARDRAIL
from naigos.research.aoi import get_aoi


def write_research_tree(staging: Path, aoi_name: str = "owens_valley", *,
                        atmosphere_payload: bytes | None = None, calls: list | None = None) -> dict:
    """What `naigos.research.run.build` + `snapshot_aoi` produce, in miniature.

    Honours the cache contract: an artifact already in the manifest is reused
    (the builder is not called), so seeded snapshots re-fetch only what was
    dropped. `calls` records which builders actually ran.
    """
    aoi = get_aoi(aoi_name)
    calls = calls if calls is not None else []

    def produce(key, source, url, rel, payload):
        def build():
            calls.append(key)
            return payload
        return cache.produce(key, source, url, rel, build)

    with roots.research_roots(cache_dir=staging / "cache", components_dir=staging / "components"):
        fp = aoi.fingerprint
        dem = produce(f"dem/{aoi.name}/30m/{fp}", "usgs_3dep",
                      "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer",
                      f"terrain/{aoi.name}_{fp}_dem_30m_utm.tif", b"tif-bytes")
        npz = produce(f"dem_npz/{aoi.name}/{fp}", "usgs_3dep", dem.url,
                      f"terrain/{aoi.name}_{fp}_dem_30m_utm.npz", b"npz-bytes")
        ap = produce("ourairports/airports", "ourairports",
                     "https://davidmegginson.github.io/ourairports-data/airports.csv",
                     "airspace/ourairports_airports.csv", b"ident,name\n")
        atm = produce(f"open_meteo/profile/{aoi.name}/{fp}", "open_meteo",
                      "https://api.open-meteo.com/v1/forecast",
                      f"atmosphere/open_meteo_{aoi.name}_{fp}_profile.json",
                      atmosphere_payload or b'{"hourly": {}}')
        fl = produce("opensky/states/2x12s", "opensky", "https://opensky-network.org/api/states/all",
                     "flights/opensky_states_2x12s.json", b'{"snapshots": []}')
        common = dict(role="r", inputs=["i"], outputs=["o"], decision="d", rationale="why")
        spec.write_component("env.aoi", source_keys=["usgs_3dep"], artifacts=[dem], parameters={
            "name": aoi.name, "fingerprint": fp, "bbox_wgs84": list(aoi.bbox)}, **common)
        spec.write_component("data.terrain_dem", source_keys=["usgs_3dep"], artifacts=[dem, npz], **common)
        spec.write_component("data.airfields", source_keys=["ourairports", "usgs_3dep"], artifacts=[ap], **common)
        spec.write_component("data.atmosphere", source_keys=["open_meteo"], artifacts=[atm], **common)
        spec.write_component("data.flight_envelope", source_keys=["opensky"], artifacts=[fl], **common)
        spec.write_component(
            "model.detection", source_keys=["radar_theory", "open_meteo"],
            invariants=["Blue never acts on a threat. These envelopes are environment, not targets."],
            caveats=["free-space model", GUARDRAIL.strip()], **common)
        spec.write_component("research.agent", source_keys=["usgs_3dep", "open_meteo"],
                             caveats=[GUARDRAIL.strip()], **common)
        spec.snapshot_aoi(aoi.name)
    (staging / "DATA.md").write_text("# DATA.md - provenance (fixture)\n")
    return {"components": ["env.aoi.json"], "artifacts": sorted(cache.load_manifest()),
            "egress_guard": None}


def fake_runner(calls: list | None = None, *, fail: bool = False, payload: bytes | None = None):
    """A `research_runner` for `snapshot.build` that never touches the network."""

    def run(*, staging, aoi, skip_flights, flight_snapshots, timeout_s):
        if fail:
            raise RuntimeError("upstream returned 503")
        return write_research_tree(staging, aoi, calls=calls, atmosphere_payload=payload)

    return run


# --- an offline stand-in for the deployed app ------------------------------------

import json  # noqa: E402
import pickle  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402

from naigos.pipeline import config as pcfg  # noqa: E402
from naigos.pipeline import coordinator, layout, leases, worker  # noqa: E402
from naigos.rl import runmeta  # noqa: E402

CLEAN_CODE = {"commit": "c0ffee" * 6 + "abcd", "dirty": False}


def metrics_for(q: float) -> dict:
    """Deterministic fake metrics, monotone in a policy's 'quality'."""
    return {
        "survival_rate": round(q, 6), "objective_rate": round(max(q - 0.05, 0.0), 6),
        "exposure_early": round(0.3 - 0.1 * q, 6), "shootdown_rate": round(0.2 * (1 - q), 6),
        "terrain_rate": 0.02, "bounds_rate": 0.05, "mean_exposure": 0.1,
    }


def quality_of(ckpt) -> float:
    with open(ckpt, "rb") as f:
        return float(pickle.load(f)["actor"]["quality"])


def fake_trainer(qualities: list, *, fail: bool = False, seen_roots: list | None = None):
    """Writes a run directory `runmeta.verify_run_dir` accepts. Quality per call from `qualities`."""

    def train(*, out_dir, meta, profile, seed, aoi, use_cbf, resume):
        from naigos.research import cache
        if seen_roots is not None:
            seen_roots.append(str(cache.cache_dir()))
        if fail:
            raise RuntimeError("CUDA error: out of memory")
        q = qualities.pop(0) if qualities else 0.8
        runmeta.write_metadata(out_dir, {**meta, "runtime": {"platform": "gpu"}})
        iters = profile["iterations"]
        (out_dir / "history.json").write_text(json.dumps([{"iter": 0}, {"iter": iters}]))
        runmeta.write_json(out_dir / "perf.json", {"device": {"platform": "gpu"},
                                                   "steady_iteration_s_median": 0.1})
        runmeta.write_json(out_dir / "theatre.json", {"aoi": aoi})
        with open(out_dir / f"ckpt_{iters:06d}.pkl", "wb") as f:
            pickle.dump({"actor": {"quality": q}}, f)
        man = runmeta.build_manifest(run_name=out_dir.name, job_id=None, profile=profile["name"],
                                     gpu="A10G", timeout_s=60, max_retries=0, checkpoint_every=1,
                                     code=meta["code"], config=meta["config"],
                                     status=runmeta.COMPLETED)
        man["last_iteration"] = iters
        runmeta.write_json(out_dir / "manifest.json", man)
        return {"iterations_done": iters}

    return train


def fake_measure(*, verifier_ok: bool = True, drop_metric: str | None = None, jitter: float = 0.0,
                 champion_error: str | None = None, seen: list | None = None):
    def measure(*, snapshot_root, plan, env_shape, candidate_ckpt, champion_ckpt):
        if seen is not None:
            seen.append({"plan": plan, "snapshot_root": str(snapshot_root), "env": env_shape})
        cand = metrics_for(quality_of(candidate_ckpt))
        cand = {k: v + jitter for k, v in cand.items()}
        if drop_metric:
            cand.pop(drop_metric)
        out = {
            "candidate": cand,
            "baselines": {"avoid_nap": metrics_for(0.55), "direct": metrics_for(0.3)},
            "n_episodes": len(plan["seeds"]) * plan["worlds_per_seed"], "device": "cpu",
            "verifier": {"episodes": plan["verifier_episodes"], "ok": verifier_ok,
                         "mismatches": [] if verifier_ok else ["t=3 agent=0: logged shootdown with no firing solution"],
                         "reports": []},
        }
        if champion_ckpt is not None:
            if champion_error:
                out["champion_error"] = champion_error
            else:
                out["champion_metrics"] = metrics_for(quality_of(champion_ckpt))
        return out

    return measure


class Clock:
    def __init__(self, t: datetime):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, **kw):
        self.t += timedelta(**kw)


class FakeModal:
    """Spawned calls queue here; `drain` runs them the way Modal's workers would."""

    def __init__(self):
        self.calls: dict = {}
        self.queue: list = []
        self.current = None

    def spawn(self, fn, payload):
        cid = f"fc-{len(self.calls) + 1:04d}"
        self.calls[cid] = {"fn": fn, "payload": json.loads(json.dumps(payload)), "state": "pending"}
        self.queue.append(cid)
        return cid

    def remote_state(self, call_id):
        return (self.calls.get(call_id) or {}).get("state")

    def spawned(self, fn=None):
        return [c for c in self.calls.values() if fn is None or c["fn"] == fn]


class Harness:
    """The whole pipeline against a temp Volume, a fake Modal and a fake clock."""

    def __init__(self, root, *, config_changes: dict | None = None,
                 start=datetime(2026, 9, 11, 6, 0, 30, tzinfo=timezone.utc), code=None):
        self.lay = layout.Layout(root)
        self.clock = Clock(start)
        self.modal = FakeModal()
        self.store = leases.FileLeaseStore(self.lay.root / "locks")
        self.leases = leases.LeaseManager(self.store, now=self.clock,
                                          remote_state=self.modal.remote_state)
        self.logs: list = []
        self.svc = worker.Services(
            lay=self.lay, leases=self.leases, code=code or dict(CLEAN_CODE),
            spawn=self.modal.spawn, remote_state=self.modal.remote_state,
            now=self.clock, call_id=lambda: self.modal.current, log=self.logs.append)
        self.qualities: list = []
        self.trainer_fail = False
        self.measure = fake_measure()
        base = pcfg.load_default()
        changes = {
            "training": {**base["training"], "require_smoke": False,
                         "nightly": {"profile": "short", "overrides": {"iterations": 2}, "seed": 0},
                         "weekly": {"enabled": True, "profile": "short",
                                    "overrides": {"iterations": 3}, "seed": 1}},
            **(config_changes or {}),
        }
        self.set_config(changes)

    def set_config(self, changes: dict):
        cur = layout.read_json(self.lay.config_current)
        base = cur or pcfg.load_default()
        doc = pcfg.next_version(base, changes, updated_by="test", now=self.clock())
        pcfg.publish(self.lay, doc, expected_version=(cur or {}).get("version"))
        return doc

    def handlers(self):
        return {
            coordinator.SNAPSHOT_FN: lambda p: worker.run_snapshot(
                self.svc, p, research_runner=fake_runner()),
            coordinator.TRAIN_FN: lambda p: worker.run_training(
                self.svc, p, trainer=fake_trainer(self.qualities, fail=self.trainer_fail),
                dispatch_eval=lambda c: coordinator.dispatch_evaluation(self.svc, c)),
            coordinator.EVALUATE_FN: lambda p: worker.run_evaluation(self.svc, p, measure=self.measure),
        }

    def drain(self, limit: int = 20):
        h = self.handlers()
        n = 0
        while self.modal.queue and n < limit:
            call_id = self.modal.queue.pop(0)
            call = self.modal.calls[call_id]
            self.modal.current = call_id
            call["state"] = "running"
            try:
                h[call["fn"]](call["payload"])
                call["state"] = "success"
            except Exception:
                call["state"] = "failed"
            finally:
                self.modal.current = None
            n += 1
        return n

    def tick(self, kind: str, **kw):
        return coordinator.tick(self.svc, kind, **kw)

    def champion_bytes(self) -> bytes | None:
        p = self.lay.champion_pointer
        return p.read_bytes() if p.exists() else None
