"""Traffic-R1 Appendix A.1 prompt construction and response parsing."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

# Paper action vocabulary (exactly four phases).
SIGNAL_ORDER = ("ETWT", "ELWL", "NTST", "NLSL")

SIGNAL_ALLOWED = {
    "ETWT": "Eastern and western through lanes",
    "ELWL": "Eastern and western left lanes",
    "NTST": "North and south through lanes",
    "NLSL": "North and south left lanes",
}

# (approach_from, movement) pairs served by each signal. Approach is where
# traffic comes from (N/E/S/W). Movement is through (T) or left (L).
SIGNAL_LANE_PAIRS = {
    "ETWT": (("E", "T"), ("W", "T")),
    "ELWL": (("E", "L"), ("W", "L")),
    "NTST": (("N", "T"), ("S", "T")),
    "NLSL": (("N", "L"), ("S", "L")),
}

APPROACH_WORD = {"N": "North", "S": "South", "E": "East", "W": "West"}

TASK_DESCRIPTION = (
    "The crossroad connects two roads: north-south and east-west, with the traffic "
    "light at their intersection. Each road is divided into two sections (e.g., north "
    "and south for the north-south road) and each section has two lanes: a through "
    "lane and a left-turn lane. Right turns are always permitted. Each lane is further "
    "divided into three segments. Segment 1 is the closest to the intersection. "
    "Segment 2 is in the middle. Segment 3 is the farthest. In a lane, there may be "
    "early queued vehicles and approaching vehicles traveling in different segments. "
    "Early queued vehicles have arrived at the intersection and await passage "
    "permission. Approaching vehicles will arrive at the intersection in the future."
)

FORMAT_INSTRUCTION = (
    "You can only choose one of the signals listed above. You FIRST think about the "
    "reasoning process for your choice as an internal monologue and then provide the "
    "final answer. Your think process MUST BE put in <think>...</think> tags. "
    "The final choice MUST BE put in \\boxed{}."
)

SYSTEM_PROMPT = "You are a helpful traffic control agent."

_BOXED_RE = re.compile(r"\\boxed\{\s*([A-Za-z]+)\s*\}")
_SIGNAL_TOKEN_RE = re.compile(r"\b(ETWT|ELWL|NTST|NLSL)\b", re.IGNORECASE)


class TrafficR1ParseError(ValueError):
    """Raised when a model response cannot be parsed into a valid signal."""


def build_observation_text(
    signal_stats: Dict[str, Dict[str, Dict[str, int]]],
    n_segments: int = 3,
) -> str:
    """
    Build the Structured Traffic Observation block.

    signal_stats[signal][side] = {
        'early_queued': int,
        'segments': [c1, c2, c3],  # approaching counts, seg1 closest
    }
    side is 'East'/'West' for E-W signals and 'North'/'South' for N-S signals.
    """
    lines: List[str] = ["Structured Traffic Observation:"]
    for signal in SIGNAL_ORDER:
        pair = SIGNAL_LANE_PAIRS[signal]
        side_a = APPROACH_WORD[pair[0][0]]
        side_b = APPROACH_WORD[pair[1][0]]
        stats = signal_stats[signal]
        a = stats[side_a]
        b = stats[side_b]
        q_a, q_b = int(a["early_queued"]), int(b["early_queued"])
        lines.append(f"Signal: {signal}")
        lines.append(f"Allowed lanes: {SIGNAL_ALLOWED[signal]}")
        lines.append(
            f"- Early queued: {q_a} ({side_a}), {q_b} ({side_b}), {q_a + q_b} (Total)"
        )
        segs_a = list(a["segments"]) + [0] * n_segments
        segs_b = list(b["segments"]) + [0] * n_segments
        for s in range(n_segments):
            ca, cb = int(segs_a[s]), int(segs_b[s])
            lines.append(
                f"- Segment {s + 1}: {ca} ({side_a}), {cb} ({side_b}), {ca + cb} (Total)"
            )
    return "\n".join(lines)


def build_user_prompt(
    signal_stats: Dict[str, Dict[str, Dict[str, int]]],
    n_segments: int = 3,
    incident_text: Optional[str] = None,
) -> str:
    """Full user prompt matching Traffic-R1 Appendix A.1 structure."""
    parts = [
        f"Task Description: {TASK_DESCRIPTION}",
        build_observation_text(signal_stats, n_segments=n_segments),
    ]
    if incident_text:
        parts.append(f"Incident Information:\n{incident_text}")
    parts.append(f"Format Instruction: {FORMAT_INSTRUCTION}")
    return "\n\n".join(parts)


def build_messages(
    signal_stats: Dict[str, Dict[str, Dict[str, int]]],
    n_segments: int = 3,
    incident_text: Optional[str] = None,
) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(
                signal_stats, n_segments=n_segments, incident_text=incident_text
            ),
        },
    ]


def parse_signal(response: str, allowed: Sequence[str] = SIGNAL_ORDER) -> str:
    """
    Extract the chosen signal from a Traffic-R1 response.

    Prefer \\boxed{SIGNAL}. Fall back to the last explicit signal token only if
    boxed is absent but a single allowed token appears after </think>.
    """
    if response is None:
        raise TrafficR1ParseError("Empty model response")
    text = str(response).strip()
    if not text:
        raise TrafficR1ParseError("Empty model response")

    allowed_set = {a.upper() for a in allowed}
    boxed = _BOXED_RE.findall(text)
    if boxed:
        choice = boxed[-1].upper()
        if choice not in allowed_set:
            raise TrafficR1ParseError(
                f"Boxed signal {choice!r} not in allowed set {sorted(allowed_set)}"
            )
        return choice

    # After think block, look for an explicit signal mention.
    after_think = text
    close = text.rfind("</think>")
    if close != -1:
        after_think = text[close + len("</think>") :]
    tokens = [t.upper() for t in _SIGNAL_TOKEN_RE.findall(after_think)]
    tokens = [t for t in tokens if t in allowed_set]
    if len(tokens) == 1:
        return tokens[0]
    if len(tokens) > 1:
        # Use the last mention after the think block.
        return tokens[-1]

    raise TrafficR1ParseError(
        "Could not parse a valid signal from model response "
        f"(expected \\boxed{{ETWT|ELWL|NTST|NLSL}}). Response tail: {text[-400:]!r}"
    )


def empty_signal_stats(n_segments: int = 3) -> Dict[str, Dict[str, Dict[str, int]]]:
    stats: Dict[str, Dict[str, Dict[str, int]]] = {}
    for signal, pair in SIGNAL_LANE_PAIRS.items():
        stats[signal] = {}
        for approach, _mov in pair:
            side = APPROACH_WORD[approach]
            stats[signal][side] = {
                "early_queued": 0,
                "segments": [0] * n_segments,
            }
    return stats
