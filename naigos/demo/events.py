"""Visual events: what the viewer may animate, and the state transition behind each one.

A tracer streaking from a threat to an aircraft is a claim. If it is drawn at a
moment the simulation did not resolve a shootdown, it is a lie told at the
highest fidelity on the screen. So every event the page can show is derived
here, from state the simulation already produced, and carries a `cause` naming
that state. Nothing here simulates a round, a missile or a flight time.

    type             emitted when                                   cause cites
    ---------------  ---------------------------------------------  ------------------------------
    detected         max detection probability rises through 0.5    RewardTerms.exposure
    lock_acquired    max track quality rises through lock_threshold EnvState.lock (update_track)
    terrain_masked   the drawn tracker ray's clearance goes negative terrain.los_clearance  [live]
    shot_down        the env resolves a kill for this aircraft      RewardTerms.shotdown
    launch_visual    paired with shot_down, same step, same threat  derived from shot_down;
                                                                    visual_only, no munition

Rules, each of which `tests/test_visual_events.py` pins:

  * Deterministic: a pure function of the arrays it is handed. Replay derives
    events from the full-resolution trace and reproduces them at the same logged
    frame every time it is played or scrubbed.
  * Read-only: inputs are copied to numpy; nothing is written back. Events
    cannot alter reward, detection, lock, action selection or verifier results
    because nothing in `naigos/env` or `naigos/rl` can import this module.
  * No repeats after loss: a sortie that has been shot down emits nothing more.
    Only a re-task (live) opens a new sortie, under a new sortie number, and
    events from the old one are never re-emitted.
  * Attribution is the simulation's: the threat behind a `shot_down` is the one
    contributing the largest term of that step's kill hazard (`firing` from
    `detection.engagement` times the kind's per-step kill probability), nearest
    first on a tie. If no threat had a firing solution the event names none and
    no launch visual is emitted -- a kill is still a kill, but there is nothing
    to draw a tracer from.

Numpy only; imports nothing from the simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: pd above which an aircraft counts as detected for the `detected` event. The
#: detection model's logistic is centred on its threshold SNR, so 0.5 is the
#: model's own "more likely seen than not".
DETECT_PD = 0.5

#: Track quality at which a threat's sensor/turret is drawn slewed toward an
#: aircraft. The HUD already turns an aircraft amber at this level; the turret
#: uses the same number so the two cannot disagree about who is being tracked.
TRACK_VISUAL_MIN = 0.25

EVENT_TYPES = ("detected", "lock_acquired", "terrain_masked", "shot_down", "launch_visual")

CAUSES = {
    "detected": "RewardTerms.exposure (max detection_probability pd over threats) rose through 0.5",
    "lock_acquired": "EnvState.lock (detection.update_track) rose through DetectionConfig.lock_threshold",
    "terrain_masked": "terrain.los_clearance < 0 on the drawn tracker ray (live frame)",
    "shot_down": "RewardTerms.shotdown == 1 (kill-hazard draw in NaigosEnv.step)",
    "launch_visual": ("visualisation of the simulated shot_down outcome at the same step; "
                      "no projectile or missile is simulated"),
}


def track_selection(lock, in_flight, threshold: float = TRACK_VISUAL_MIN):
    """(T,) index of the aircraft each threat holds its best track on, or -1.

    Straight off the lock matrix: the argmax over aircraft still flying, kept
    only if that track reaches `threshold`. This is what a turret is allowed to
    point at.
    """
    lock = np.asarray(lock, dtype=np.float64)
    live = np.asarray(in_flight, dtype=bool)
    masked = np.where(live[None, :], lock, -1.0)
    best = masked.argmax(axis=1)
    ok = masked[np.arange(lock.shape[0]), best] >= threshold
    return np.where(ok, best, -1).astype(int)


def attribute_kill(firing, kill_weight, blue_pos, threat_pos, b: int) -> int | None:
    """The threat behind aircraft `b`'s kill this step, or None if none was firing."""
    f = np.asarray(firing, dtype=np.float64)[:, b]
    if not (f > 0).any():
        return None
    w = np.ones_like(f) if kill_weight is None else np.asarray(kill_weight, dtype=np.float64)
    term = f * w
    rng = np.linalg.norm(np.asarray(threat_pos, dtype=np.float64)
                         - np.asarray(blue_pos, dtype=np.float64)[b], axis=-1)
    # largest hazard term; nearest breaks ties; lowest index breaks those
    order = np.lexsort((np.arange(len(f)), rng, -term))
    top = int(order[0])
    return top if term[top] > 0 else None


@dataclass
class EventDeriver:
    """Turns successive simulation steps into visual events. Holds only its own flags.

    One instance per stream: the live server keeps one for the session, replay
    builds one per recorded policy. `step()` is called once per env step with
    that step's post-step arrays.
    """

    n_blue: int
    lock_threshold: float
    detect_pd: float = DETECT_PD
    _sortie: np.ndarray = field(init=False)
    _closed: np.ndarray = field(init=False)
    _detected: np.ndarray = field(init=False)
    _locked: np.ndarray = field(init=False)
    _masked: np.ndarray = field(init=False)

    def __post_init__(self):
        self._sortie = np.zeros(self.n_blue, dtype=int)
        self._closed = np.zeros(self.n_blue, dtype=bool)
        self._detected = np.zeros(self.n_blue, dtype=bool)
        self._locked = np.zeros(self.n_blue, dtype=bool)
        self._masked = np.zeros(self.n_blue, dtype=bool)

    def _reset(self, mask):
        for name in ("_detected", "_locked", "_masked"):
            setattr(self, name, getattr(self, name) & ~mask)

    def retask(self, mask) -> None:
        """Open a new sortie for each masked aircraft (the live server's respawn)."""
        mask = np.asarray(mask, dtype=bool)
        self._sortie = self._sortie + mask
        self._closed = self._closed & ~mask
        self._reset(mask)

    def step(self, *, step: int, t_s: float, in_flight, alive, shotdown, lock, exposure,
             blue_pos, threat_pos, firing=None, kill_weight=None, masked=None) -> list[dict]:
        """Events for one env step. Every argument is read, none is modified.

        `in_flight` is whether each aircraft was flying at the START of the step
        (alive and not arrived); only those can generate events.
        """
        in_flight = np.asarray(in_flight, dtype=bool) & ~self._closed
        shot = np.asarray(shotdown, dtype=bool) & in_flight
        lock = np.asarray(lock, dtype=np.float64)
        best_lock = lock.max(axis=0) if lock.size else np.zeros(self.n_blue)
        exposed = np.asarray(exposure, dtype=np.float64) >= self.detect_pd
        locked = best_lock >= self.lock_threshold
        masked_now = (np.zeros(self.n_blue, dtype=bool) if masked is None
                      else np.asarray(masked, dtype=bool))
        blue_pos = np.asarray(blue_pos, dtype=np.float64)
        threat_pos = np.asarray(threat_pos, dtype=np.float64)

        out = []

        def emit(kind, b, threat=None):
            out.append({
                "id": f"{b}.{int(self._sortie[b])}.{kind}.{int(step)}",
                "type": kind,
                "aircraft": int(b),
                "sortie": int(self._sortie[b]),
                "step": int(step),
                "t_s": round(float(t_s), 3),
                "threat": None if threat is None else int(threat),
                "pos": [round(float(v), 1) for v in blue_pos[b]],
                "threat_pos": (None if threat is None
                               else [round(float(v), 1) for v in threat_pos[threat]]),
                "cause": CAUSES[kind],
                "visual_only": kind == "launch_visual",
            })

        for b in np.flatnonzero(in_flight):
            if exposed[b] and not self._detected[b]:
                emit("detected", b)
            if locked[b] and not self._locked[b]:
                emit("lock_acquired", b, int(lock[:, b].argmax()))
            if masked is not None and masked_now[b] and not self._masked[b]:
                emit("terrain_masked", b)
            if shot[b]:
                who = (attribute_kill(firing, kill_weight, blue_pos, threat_pos, b)
                       if firing is not None else None)
                emit("shot_down", b, who)
                if who is not None:
                    emit("launch_visual", b, who)

        live_after = np.asarray(alive, dtype=bool)
        # flags follow the state for aircraft still flying; a loss closes the
        # sortie so nothing more can be emitted for it
        fly = in_flight & live_after
        self._detected = np.where(fly, exposed, self._detected)
        self._locked = np.where(fly, locked, self._locked)
        self._masked = np.where(fly, masked_now, self._masked)
        self._closed = self._closed | (in_flight & ~live_after) | shot
        return out


def replay_events(trace: dict, *, lock_threshold: float, stride: int, frame_dt: float,
                  kill_weight=None) -> list[dict]:
    """Events for one recorded rollout, keyed to the logged frame that shows them.

    `trace` holds full-resolution (unstrided) numpy arrays straight from
    `env.rollout`: alive (S,B), reached (S,B), shotdown (S,B), exposure (S,B),
    lock (S,T,B), pos (S,B,3), threat_pos (S,T,3) and, when recorded, firing
    (S,T,B). Step s is the transition that produced state s.

    The recording keeps every `stride`-th state, so an event at step s first
    appears at logged frame ceil(s / stride) -- the first frame at or after it,
    which for a kill is the frame whose `alive` is first False. That frame
    index is what the page keys the effect to, so a replay shows it at the same
    frame on every play and every scrub.
    """
    alive = np.asarray(trace["alive"], dtype=bool)
    reached = np.asarray(trace["reached"], dtype=bool)
    S, B = alive.shape
    n_frames = (S + stride - 1) // stride
    firing = trace.get("firing")
    der = EventDeriver(n_blue=B, lock_threshold=lock_threshold)
    events = []
    for s in range(S):
        if s == 0:
            in_flight = np.ones(B, dtype=bool)
        else:
            in_flight = alive[s - 1] & ~reached[s - 1]
        evs = der.step(
            step=s, t_s=0.0, in_flight=in_flight, alive=alive[s],
            shotdown=np.asarray(trace["shotdown"])[s] > 0.5,
            lock=np.asarray(trace["lock"])[s], exposure=np.asarray(trace["exposure"])[s],
            blue_pos=np.asarray(trace["pos"])[s], threat_pos=np.asarray(trace["threat_pos"])[s],
            firing=None if firing is None else np.asarray(firing)[s],
            kill_weight=kill_weight,
        )
        frame = min(n_frames - 1, -(-s // stride))
        for ev in evs:
            ev["frame"] = int(frame)
            ev["t_s"] = round(frame * frame_dt, 3)
        events += evs
    return events
