"""Validate the short/long request experiment before applying its latency gates.

All intervals must use one monotonic clock domain. The caller owns request
execution, worker supervision and the final snapshot of every attempted request.
This module assesses that snapshot without threads, network calls or GPU imports.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RequestSample:
    ok: bool
    started_s: float
    finished_s: float
    elapsed_s: float
    error: str | None = None


def _number(value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _successful(
    samples: Sequence[RequestSample], label: str, problems: list[str]
) -> list[RequestSample]:
    """Check every attempt, returning only successful samples with usable clocks."""
    successful = []
    for index, sample in enumerate(samples):
        location = f"{label}[{index}]"
        before = len(problems)
        if not isinstance(sample, RequestSample):
            problems.append(f"{location} must be a RequestSample")
            continue
        if sample.ok is not True:
            problems.append(f"{location} did not succeed")
        if sample.error is not None:
            problems.append(f"{location} contains a request error")
        clocks_valid = True
        for field in ("started_s", "finished_s", "elapsed_s"):
            value = getattr(sample, field)
            if not _number(value):
                problems.append(f"{location}.{field} must be finite")
                clocks_valid = False
            elif value < 0 or (field != "started_s" and value == 0):
                problems.append(f"{location}.{field} must be positive (started_s may be zero)")
                clocks_valid = False
        if clocks_valid and sample.finished_s <= sample.started_s:
            problems.append(f"{location}.finished_s must be after started_s")
        if len(problems) == before:
            successful.append(sample)
    return successful


def _upper_median(samples: Sequence[RequestSample]) -> float | None:
    values = sorted(float(sample.elapsed_s) for sample in samples)
    return values[len(values) // 2] if values else None


def assess_load(
    solo: Sequence[RequestSample],
    loaded: Sequence[RequestSample],
    long_samples: Sequence[RequestSample],
    *,
    expected_count: int,
    worker_stopped: bool = True,
    worker_error: str | None = None,
) -> dict:
    """Return validity and the unchanged no-barrier latency gate.

    Every attempt must succeed, and every loaded short request must have positive
    interval overlap with a successful long request. Only long requests actually
    overlapping loaded work enter its upper median. Metrics are diagnostic on an
    invalid experiment; ``passed`` can only be true when ``valid`` is true.
    Empty or non-finite derived metrics are None, never NaN/Infinity.

    ``elapsed_s`` may measure the request itself while its interval includes
    caller overhead; equality with ``finished_s - started_s`` is not required.
    """
    problems = []
    groups = {}
    for label, samples in (("solo", solo), ("loaded", loaded), ("long", long_samples)):
        if isinstance(samples, Sequence) and not isinstance(samples, (str, bytes)):
            groups[label] = samples
        else:
            problems.append(f"{label} must be a sequence of RequestSample objects")
            groups[label] = []

    count_valid = isinstance(expected_count, int) and not isinstance(expected_count, bool) and expected_count > 0
    if not count_valid:
        problems.append("expected_count must be a positive integer")
    for label in ("solo", "loaded"):
        if not groups[label]:
            problems.append(f"{label} must contain attempted requests")
        if count_valid and len(groups[label]) != expected_count:
            problems.append(f"{label} has {len(groups[label])} attempts; expected {expected_count}")
    if not groups["long"]:
        problems.append("long must contain at least one attempted request")
    if worker_stopped is not True:
        problems.append("long request worker has not stopped")
    if worker_error is not None:
        problems.append(f"long request worker failed: {worker_error}")

    valid_solo = _successful(groups["solo"], "solo", problems)
    valid_loaded = _successful(groups["loaded"], "loaded", problems)
    valid_long = _successful(groups["long"], "long", problems)
    overlapping_long = set()
    for index, short in enumerate(valid_loaded):
        overlaps = {
            long_index
            for long_index, long in enumerate(valid_long)
            if max(short.started_s, long.started_s) < min(short.finished_s, long.finished_s)
        }
        if not overlaps:
            problems.append(f"successful loaded request {index} has no positive overlap with a successful long request")
        overlapping_long.update(overlaps)

    solo_p50 = _upper_median(valid_solo)
    loaded_p50 = _upper_median(valid_loaded)
    long_p50 = _upper_median([valid_long[index] for index in sorted(overlapping_long)])
    ratio = loaded_p50 / solo_p50 if loaded_p50 is not None and solo_p50 is not None else None
    if ratio is not None and not math.isfinite(ratio):
        problems.append("loaded/solo latency ratio is not finite")
        ratio = None
    valid = not problems
    passed = (
        valid
        and ratio is not None
        and long_p50 is not None
        and loaded_p50 is not None
        and ratio <= 1.5
        and loaded_p50 < 0.5 * long_p50
    )
    return {
        "valid": valid,
        "passed": passed,
        "problems": problems,
        "solo_p50_s": solo_p50,
        "loaded_p50_s": loaded_p50,
        "long_p50_s": long_p50,
        "ratio": ratio,
        "overlapping_long_count": len(overlapping_long),
    }
