"""Five-field cron expressions, evaluated in UTC.

Modal fires the deployed ``modal.Cron`` schedules; this module exists for the
two things the pipeline has to know about a schedule without asking Modal:

  * **which window a tick belongs to.** Every stage's idempotency key contains
    the most recent scheduled fire time at or before "now". Two invocations of
    the same cron -- a duplicate delivery, or an old invocation still running
    when a manual "run now" lands -- compute the same window, hence the same
    key, hence the same snapshot or candidate ID, and the second one skips.
  * **when the next run is**, so ``scripts/pipeline.py status`` can report a
    stage as *scheduled* with a time, from a fresh clone, offline.

Supported syntax per field: ``*``, ``N``, ``A-B``, ``A,B,...`` and ``/step`` on
any of those. Day-of-week is 0-6 with 0 (or 7) = Sunday. When both day-of-month
and day-of-week are restricted, either matching is enough -- standard cron.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

_FIELDS = (("minute", 0, 59), ("hour", 0, 23), ("day", 1, 31), ("month", 1, 12), ("weekday", 0, 7))


class CronError(ValueError):
    """A cron expression this module will not guess the meaning of."""


@dataclass(frozen=True)
class Cron:
    expr: str
    minutes: frozenset
    hours: frozenset
    days: frozenset
    months: frozenset
    weekdays: frozenset
    day_restricted: bool
    weekday_restricted: bool

    def matches(self, t: datetime) -> bool:
        if t.minute not in self.minutes or t.hour not in self.hours or t.month not in self.months:
            return False
        return self._day_ok(t)

    def _day_ok(self, t: datetime) -> bool:
        dom = t.day in self.days
        dow = ((t.weekday() + 1) % 7) in self.weekdays  # python Monday=0 -> cron Sunday=0
        if self.day_restricted and self.weekday_restricted:
            return dom or dow
        return dom and dow


def _parse_field(text: str, lo: int, hi: int, name: str) -> tuple[frozenset, bool]:
    values: set[int] = set()
    restricted = text != "*"
    for part in text.split(","):
        step = 1
        if "/" in part:
            part, step_s = part.split("/", 1)
            if not step_s.isdigit() or int(step_s) < 1:
                raise CronError(f"{name}: bad step {step_s!r}")
            step = int(step_s)
        if part == "*":
            start, end = lo, hi
        elif "-" in part:
            a, b = part.split("-", 1)
            if not (a.isdigit() and b.isdigit()):
                raise CronError(f"{name}: bad range {part!r}")
            start, end = int(a), int(b)
        elif part.isdigit():
            start = end = int(part)
            if step != 1:
                end = hi
        else:
            raise CronError(f"{name}: cannot parse {part!r}")
        if start < lo or end > hi or start > end:
            raise CronError(f"{name}: {part!r} outside {lo}-{hi}")
        values.update(range(start, end + 1, step))
    if not values:
        raise CronError(f"{name}: matches nothing")
    return frozenset(values), restricted


def parse_cron(expr: str) -> Cron:
    if not isinstance(expr, str):
        raise CronError(f"cron expression must be a string, got {expr!r}")
    parts = expr.split()
    if len(parts) != 5:
        raise CronError(f"{expr!r}: expected 5 fields (minute hour day month weekday)")
    parsed = [_parse_field(p, lo, hi, name) for p, (name, lo, hi) in zip(parts, _FIELDS)]
    weekdays = frozenset(0 if d == 7 else d for d in parsed[4][0])
    return Cron(
        expr=expr,
        minutes=parsed[0][0],
        hours=parsed[1][0],
        days=parsed[2][0],
        months=parsed[3][0],
        weekdays=weekdays,
        day_restricted=parsed[2][1],
        weekday_restricted=parsed[4][1],
    )


def _utc_minute(t: datetime) -> datetime:
    if t.tzinfo is None:
        raise CronError("naive datetime; pass an aware UTC datetime")
    return t.astimezone(timezone.utc).replace(second=0, microsecond=0)


def _scan(cron: Cron, start: datetime, direction: int) -> datetime:
    """Walk day by day, then hour, then minute. Bounded at ~5 years of days."""
    day = start.replace(hour=0, minute=0)
    for _ in range(366 * 5):
        if day.month in cron.months and cron._day_ok(day):
            hours = sorted(cron.hours, reverse=direction < 0)
            minutes = sorted(cron.minutes, reverse=direction < 0)
            for h in hours:
                for m in minutes:
                    cand = day.replace(hour=h, minute=m)
                    if (direction > 0 and cand >= start) or (direction < 0 and cand <= start):
                        return cand
        day = day + timedelta(days=direction)
    raise CronError(f"{cron.expr!r} does not fire within five years")


def next_fire(expr: str | Cron, after: datetime) -> datetime:
    """The first scheduled time strictly after ``after``."""
    cron = expr if isinstance(expr, Cron) else parse_cron(expr)
    return _scan(cron, _utc_minute(after) + timedelta(minutes=1), +1)


def previous_fire(expr: str | Cron, at: datetime) -> datetime:
    """The most recent scheduled time at or before ``at``."""
    cron = expr if isinstance(expr, Cron) else parse_cron(expr)
    return _scan(cron, _utc_minute(at), -1)


def window_label(fire: datetime) -> str:
    """The date part used in IDs: ``YYYYMMDD`` of the scheduled fire time."""
    return fire.astimezone(timezone.utc).strftime("%Y%m%d")


def manual_window(now: datetime) -> str:
    """A manual (operator-triggered) run is its own window, stamped to the second."""
    return now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
