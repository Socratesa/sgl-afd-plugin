"""Validated alignment metrics for acceptance results; no server or GPU imports.

Rows contain a generated ``token``, ``top`` pairs of [logprob, token_id], and
``prompt_lp``. New collections also carry an ``input_id`` and a parallel list of
``prompt_positions``. Legacy rows can be compared structurally, but their input
identity cannot be verified. Callers must check ``valid`` before applying gates
and require ``identity_verified`` when using results as formal acceptance data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class _Row:
    token: int
    top: dict[int, float]
    prompt_lp: list[float]
    input_id: str | None
    positions: list[int] | None


def _token_id(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _finite(value: object, location: str, problems: list[str]) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            number = float(value)
        except (OverflowError, ValueError):
            number = math.inf
        if math.isfinite(number):
            return number
    problems.append(f"{location} must be a finite number")
    return None


def _row(value: object, location: str, expected_topk: int, problems: list[str]) -> _Row | None:
    before = len(problems)
    if not isinstance(value, dict):
        problems.append(f"{location} must be an object")
        return None
    if "error" in value:
        problems.append(f"{location} contains a request error")
        return None

    token = value.get("token")
    if not _token_id(token):
        problems.append(f"{location}.token must be a nonnegative integer")

    top = {}
    entries = value.get("top")
    if not isinstance(entries, (list, tuple)):
        problems.append(f"{location}.top must be a list of logprob/token pairs")
    else:
        if len(entries) != expected_topk:
            problems.append(f"{location}.top has {len(entries)} entries; expected {expected_topk}")
        for index, entry in enumerate(entries):
            field = f"{location}.top[{index}]"
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                problems.append(f"{field} must be a logprob/token pair")
                continue
            logprob = _finite(entry[0], f"{field}.logprob", problems)
            token_id = entry[1]
            if not _token_id(token_id):
                problems.append(f"{field}.token must be a nonnegative integer")
            elif token_id in top:
                problems.append(f"{location}.top contains duplicate token {token_id}")
            else:
                top[token_id] = logprob

    prompt = []
    raw_prompt = value.get("prompt_lp")
    if not isinstance(raw_prompt, (list, tuple)) or not raw_prompt:
        problems.append(f"{location}.prompt_lp must be a nonempty list")
    else:
        prompt = [
            _finite(logprob, f"{location}.prompt_lp[{index}]", problems)
            for index, logprob in enumerate(raw_prompt)
        ]

    input_id = value.get("input_id")
    if "input_id" in value and (not isinstance(input_id, str) or not input_id):
        problems.append(f"{location}.input_id must be a nonempty string when present")

    positions = value.get("prompt_positions")
    if "prompt_positions" in value:
        if not isinstance(positions, (list, tuple)):
            problems.append(f"{location}.prompt_positions must be a list when present")
        elif any(not _token_id(position) for position in positions):
            problems.append(f"{location}.prompt_positions must contain nonnegative integers")
        else:
            if len(positions) != len(prompt):
                problems.append(f"{location}.prompt_positions length differs from prompt_lp")
            if len(set(positions)) != len(positions):
                problems.append(f"{location}.prompt_positions contains duplicate positions")
            positions = list(positions)

    if len(problems) != before:
        return None
    return _Row(token, top, prompt, input_id, positions)


def compare(
    ref: list[dict],
    got: list[dict],
    *,
    expected_count: int | None = None,
    expected_topk: int = 5,
) -> dict:
    """Compare complete rows and report coverage as well as numerical differences.

    ``errors`` counts explicit error rows from both inputs. ``n`` remains the
    reference length. Metrics cover only individually valid, aligned row pairs;
    any malformed or missing row makes the whole result invalid. Empty metrics
    are None, never NaN. Invalid expected-count/top-k arguments raise ValueError.

    Complete top-k sets may legitimately differ across deployments: this keeps
    ``valid`` true, counts affected pairs in ``topk_set_mismatches``, and reports
    the total token symmetric difference in ``topk_symmetric_difference``.
    ``max_d_out`` covers only the reported ``compared_topk_entries`` common ids.
    ``exact`` requires equal generated tokens, full top-k maps and every prompt
    logprob at the same position; it does not imply verified legacy identity.
    """
    if not _token_id(expected_topk) or expected_topk == 0:
        raise ValueError("expected_topk must be a positive integer")
    if expected_count is not None and (not _token_id(expected_count) or expected_count == 0):
        raise ValueError("expected_count must be a positive integer or None")

    problems = []
    if not isinstance(ref, list):
        problems.append("ref must be a list")
        ref = []
    if not isinstance(got, list):
        problems.append("got must be a list")
        got = []
    if not ref or not got:
        problems.append("ref and got must both contain results")
    if len(ref) != len(got):
        problems.append(f"result counts differ: ref={len(ref)}, got={len(got)}")
    if expected_count is not None:
        for label, rows in (("ref", ref), ("got", got)):
            if len(rows) != expected_count:
                problems.append(f"{label} has {len(rows)} results; expected {expected_count}")

    errors = sum(isinstance(row, dict) and "error" in row for rows in (ref, got) for row in rows)
    reference = [_row(row, f"ref[{index}]", expected_topk, problems) for index, row in enumerate(ref)]
    candidate = [_row(row, f"got[{index}]", expected_topk, problems) for index, row in enumerate(got)]
    first_token_match = 0
    set_mismatches = 0
    symmetric_difference = 0
    output_deltas = []
    prompt_deltas = []
    identity_verified = bool(ref) and len(ref) == len(got)

    for index in range(min(len(reference), len(candidate))):
        left, right = reference[index], candidate[index]
        if left is None or right is None:
            identity_verified = False
            continue
        before = len(problems)
        if left.input_id != right.input_id:
            problems.append(f"row {index}: input_id mismatch or missing on one side")
        if left.positions != right.positions:
            problems.append(f"row {index}: prompt_positions mismatch or missing on one side")
        if len(left.prompt_lp) != len(right.prompt_lp):
            problems.append(f"row {index}: prompt_lp lengths differ")
        if left.input_id is None or right.input_id is None or left.positions is None or right.positions is None:
            identity_verified = False
        if len(problems) != before:
            identity_verified = False
            continue

        common = left.top.keys() & right.top.keys()
        output = [abs(left.top[token] - right.top[token]) for token in sorted(common)]
        prompt = [abs(left.prompt_lp[position] - right.prompt_lp[position]) for position in range(len(left.prompt_lp))]
        if any(not math.isfinite(delta) for delta in output + prompt):
            problems.append(f"row {index}: comparison produced a non-finite delta")
            continue
        changed = left.top.keys() ^ right.top.keys()
        set_mismatches += bool(changed)
        symmetric_difference += len(changed)
        first_token_match += left.token == right.token
        output_deltas.extend(output)
        prompt_deltas.extend(prompt)

    prompt_deltas.sort()
    maximum_output = max(output_deltas, default=None)
    maximum_prompt = max(prompt_deltas, default=None)
    mean = maximum_prompt
    if maximum_prompt:
        # Scaling before summing keeps even very large finite deltas finite.
        mean = (math.fsum(delta / maximum_prompt for delta in prompt_deltas) / len(prompt_deltas)) * maximum_prompt
    p99 = prompt_deltas[min(len(prompt_deltas) - 1, int(0.99 * len(prompt_deltas)))] if prompt_deltas else None
    valid = not problems
    exact = (
        valid
        and first_token_match == len(ref)
        and set_mismatches == 0
        and maximum_output == 0
        and maximum_prompt == 0
    )
    return {
        "n": len(ref),
        "errors": errors,
        "first_token_match": first_token_match,
        "max_d_out": maximum_output,
        "mean_d_prompt": mean,
        "p99_d_prompt": p99,
        "max_d_prompt": maximum_prompt,
        "prompt_positions": len(prompt_deltas),
        "valid": valid,
        "problems": problems,
        "exact": exact,
        "identity_verified": valid and identity_verified,
        "topk_set_mismatches": set_mismatches,
        "topk_symmetric_difference": symmetric_difference,
        "compared_topk_entries": len(output_deltas),
    }
