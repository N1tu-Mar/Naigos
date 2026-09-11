"""Leases and job status: one writer per stage, and ambiguity fails closed.

Offline. The Modal Dict backend is exercised through a fake with the documented
`put(key, value, skip_if_exists=True) -> bool` contract.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

import pytest

from naigos.pipeline import jobs, layout, leases

T0 = datetime(2026, 9, 11, 8, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, **kw):
        self.t += timedelta(**kw)


class FakeModalDict:
    """The documented modal.Dict surface a lease needs, with an atomic put."""

    def __init__(self):
        self._d = {}
        self._lock = threading.Lock()

    def put(self, key, value, *, skip_if_exists=False):
        with self._lock:
            if skip_if_exists and key in self._d:
                return False
            self._d[key] = value
            return True

    def get(self, key, default=None):
        return self._d.get(key, default)

    def pop(self, key, *default):
        with self._lock:
            if key not in self._d and not default:
                raise KeyError(key)
            return self._d.pop(key, *default)


@pytest.fixture(params=["file", "dict"])
def store(request, tmp_path):
    if request.param == "file":
        return leases.FileLeaseStore(tmp_path / "locks")
    return leases.DictLeaseStore(FakeModalDict(), mirror_dir=tmp_path / "locks")


KEY = "candidate:c-20260911-nightly-0123456789ab"


def test_one_writer_per_candidate(store):
    clock = Clock()
    mgr = leases.LeaseManager(store, now=clock)
    first = mgr.acquire(KEY, "train-worker-a")
    with pytest.raises(leases.LeaseHeld):
        mgr.acquire(KEY, "train-worker-b")
    mgr.release(first)
    second = mgr.acquire(KEY, "train-worker-b")
    assert second.token != first.token


def test_concurrent_acquire_admits_exactly_one(store):
    mgr = leases.LeaseManager(store, now=Clock())
    winners, losers = [], []
    barrier = threading.Barrier(8)

    def attempt(i):
        barrier.wait()
        try:
            winners.append(mgr.acquire(KEY, f"w{i}"))
        except leases.LeaseHeld:
            losers.append(i)

    threads = [threading.Thread(target=attempt, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(winners) == 1 and len(losers) == 7


def test_a_stale_lease_with_no_remote_answer_fails_closed(store):
    clock = Clock()
    mgr = leases.LeaseManager(store, now=clock)
    held = mgr.acquire(KEY, "train-worker-a")
    mgr.heartbeat(held, call_id="fc-dead-or-alive")
    clock.advance(seconds=leases.STALE_AFTER_S + 1)
    with pytest.raises(leases.LeaseAmbiguous) as e:
        mgr.acquire(KEY, "train-worker-b")
    assert "Refusing to guess" in str(e.value)
    assert store.get(KEY)["token"] == held.token  # untouched


def test_a_lease_that_never_recorded_a_call_is_ambiguous_when_stale(store):
    clock = Clock()
    mgr = leases.LeaseManager(store, now=clock)
    mgr.acquire(KEY, "coordinator")  # died before spawning
    clock.advance(hours=3)
    with pytest.raises(leases.LeaseAmbiguous, match="never recorded a call"):
        mgr.acquire(KEY, "coordinator-2")


def test_a_lease_whose_call_is_terminal_is_reclaimed(store):
    clock = Clock()
    remote = {"fc-1": "FunctionTimeoutError"}
    mgr = leases.LeaseManager(store, now=clock, remote_state=lambda cid: remote.get(cid))
    held = mgr.acquire(KEY, "a")
    mgr.heartbeat(held, call_id="fc-1")
    new = mgr.acquire(KEY, "b")  # Modal says fc-1 timed out: no writer can exist
    assert store.get(KEY)["token"] == new.token
    assert store.get(KEY)["reclaimed_from"]["token"] == held.token
    with pytest.raises(leases.LeaseLost):
        mgr.heartbeat(held)


def test_a_live_call_holds_its_lease_even_with_an_old_heartbeat(store):
    clock = Clock()
    mgr = leases.LeaseManager(store, now=clock, remote_state=lambda cid: "running")
    held = mgr.acquire(KEY, "a")
    mgr.heartbeat(held, call_id="fc-2")
    clock.advance(hours=5)
    with pytest.raises(leases.LeaseHeld):
        mgr.acquire(KEY, "b")


def test_release_of_someone_elses_lease_is_a_noop(store):
    clock = Clock()
    remote = {"fc-1": "failed"}
    mgr = leases.LeaseManager(store, now=clock, remote_state=lambda cid: remote.get(cid))
    old = mgr.acquire(KEY, "a")
    mgr.heartbeat(old, call_id="fc-1")
    new = mgr.acquire(KEY, "b")
    mgr.release(old)
    assert store.get(KEY)["token"] == new.token


def test_force_clear_is_the_operator_escape_hatch(store):
    mgr = leases.LeaseManager(store, now=Clock())
    held = mgr.acquire(KEY, "a")
    removed = mgr.force_clear(KEY)
    assert removed["token"] == held.token
    assert store.get(KEY) is None


def test_keep_alive_heartbeats_and_notices_a_lost_lease(tmp_path):
    store = leases.FileLeaseStore(tmp_path)
    mgr = leases.LeaseManager(store)
    lease = mgr.acquire(KEY, "a")
    first_beat = store.get(KEY)["heartbeat_utc"]
    with mgr.keep_alive(lease, interval_s=0.01) as state:
        import time
        time.sleep(0.05)
        mgr.force_clear(KEY)
        time.sleep(0.05)
    assert state["lost"] and "no longer holds" in state["lost"]
    assert first_beat


def test_unreadable_lease_is_not_free(tmp_path):
    store = leases.FileLeaseStore(tmp_path)
    path = store._path(KEY)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{garbage")
    with pytest.raises(leases.LeaseAmbiguous):
        leases.LeaseManager(store).acquire(KEY, "a")


def test_dict_store_mirrors_to_the_volume(tmp_path):
    store = leases.DictLeaseStore(FakeModalDict(), mirror_dir=tmp_path / "locks")
    mgr = leases.LeaseManager(store)
    lease = mgr.acquire(KEY, "a")
    mirror = tmp_path / "locks" / "candidate__c-20260911-nightly-0123456789ab.json"
    assert mirror.exists()
    mgr.release(lease)
    assert not mirror.exists()


# --- job status ----------------------------------------------------------------


def _record(status=jobs.QUEUED, **kw):
    rec = jobs.new_record(stage="train", target="c-20260911-nightly-0123456789ab",
                          idempotency_key="k" * 64, window="20260911", config_digest="d",
                          code={"commit": "abc"}, parents={}, now=T0)
    return jobs.update(rec, status=status, now=T0, **kw) if status != jobs.QUEUED else rec


def test_worker_recorded_terminal_status_wins():
    v = jobs.classify(_record(jobs.COMPLETED), remote_state="failed", now=T0)
    assert v["status"] == jobs.COMPLETED


def test_upstream_terminal_overrides_a_live_record():
    v = jobs.classify(_record(jobs.RUNNING), remote_state="FunctionTimeoutError", now=T0)
    assert v["status"] == jobs.FAILED and "lost" in v["failure_reason"]
    v = jobs.classify(_record(jobs.RUNNING), remote_state="success", now=T0)
    assert v["status"] == jobs.FAILED and "not trusted" in v["failure_reason"]


def test_stale_live_record_with_no_remote_answer_is_unknown():
    v = jobs.classify(_record(jobs.RUNNING), now=T0 + timedelta(hours=2))
    assert v["status"] == jobs.UNKNOWN
    v = jobs.classify(_record(jobs.RUNNING), remote_state="running", now=T0 + timedelta(hours=2))
    assert v["status"] == jobs.RUNNING


def test_queued_is_distinct_from_running():
    assert jobs.classify(_record(), remote_state="pending", now=T0)["status"] == jobs.QUEUED
    assert jobs.classify(_record(), remote_state="running", now=T0)["status"] == jobs.RUNNING


def test_job_updates_redact_and_append_events(tmp_path):
    lay = layout.Layout(tmp_path)
    secret = "as-" + "A1b2C3d4E5f6G7h8"
    rec = jobs.update(_record(), status=jobs.FAILED, failure_reason=f"boom {secret}", now=T0)
    jobs.write(lay, rec)
    back = jobs.read(lay, rec["key"])
    assert secret not in back["failure_reason"]
    assert [e["status"] for e in back["events"]] == [jobs.QUEUED, jobs.FAILED]
    assert jobs.all_records(lay)[0]["key"] == rec["key"]
    many = rec
    for _ in range(80):
        many = jobs.update(many, status=jobs.RUNNING, now=T0)
    assert len(many["events"]) == 50


def test_status_vocabulary_is_the_documented_one():
    assert set(jobs.STATUSES) >= {"scheduled", "queued", "running", "completed", "failed",
                                  "skipped", "rejected", "inconclusive", "promoted"}
