"""
Traffic-R1 agent for LibSignal (SUMO).

Faithful inference-only integration of Season998/Traffic-R1:
- Appendix A.1 four-phase prompt (ETWT / ELWL / NTST / NLSL)
- Independent per-intersection decisions (no async messaging in v1)
- Shared backend (local HF or OpenAI-compatible API)
- Parse failures: retry up to parse_retries, then hard-fail
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from agent.base import BaseAgent
from agent.traffic_r1_backend import TrafficR1Backend, create_backend
from agent.traffic_r1_prompt import (
    APPROACH_WORD,
    SIGNAL_LANE_PAIRS,
    SIGNAL_ORDER,
    TrafficR1ParseError,
    build_messages,
    empty_signal_stats,
    parse_signal,
)
from common.registry import Registry
from generator import IntersectionPhaseGenerator, LaneVehicleGenerator

# Shared across all TrafficR1Agent instances in one process.
_SHARED_BACKEND: Optional[TrafficR1Backend] = None


def _get_shared_backend(param: Dict[str, Any]) -> TrafficR1Backend:
    global _SHARED_BACKEND
    if _SHARED_BACKEND is None:
        _SHARED_BACKEND = create_backend(param)
    return _SHARED_BACKEND


def reset_shared_backend() -> None:
    """Test helper to clear the process-wide backend singleton."""
    global _SHARED_BACKEND
    _SHARED_BACKEND = None


def _compass_from_atan2(angle: float) -> str:
    """Map Intersection._get_direction atan2(x,y) angle to N/E/S/W."""
    deg = (angle * 180.0 / math.pi) % 360.0
    if deg < 45.0 or deg >= 315.0:
        return "N"
    if 45.0 <= deg < 135.0:
        return "E"
    if 135.0 <= deg < 225.0:
        return "S"
    return "W"


@Registry.register_model("traffic_r1")
class TrafficR1Agent(BaseAgent):
    """Traffic-R1 LLM controller (test / zero-shot only)."""

    def __init__(self, world, rank):
        super().__init__(world)
        self.world = world
        self.rank = rank
        self.model = None

        param = Registry.mapping["model_mapping"]["setting"].param
        self.param = param
        self.v_stop = float(param.get("v_stop", 0.1))
        self.n_segments = int(param.get("n_segments", 3))
        self.parse_retries = int(param.get("parse_retries", 5))
        self.obs_distance = float(
            param.get("obs_distance", getattr(world, "max_distance", 200.0))
        )

        inter_id = self.world.intersection_ids[self.rank]
        self.inter_obj = self.world.id2intersection[inter_id]
        self._init_generators()

        self.action_space = gym.spaces.Discrete(len(self.inter_obj.phases))
        self._lane_meta = self._build_lane_meta()
        self._signal_to_phase = self._resolve_phase_map(param)
        self._llm_enabled = len(self.inter_obj.phases) >= 4 and bool(
            self._signal_to_phase
        )

        # Lazy: only intersections that need the LLM touch the backend.
        self._backend: Optional[TrafficR1Backend] = None
        self.last_action = 0
        self.last_raw_response: Optional[str] = None
        self.last_signal: Optional[str] = None

    def _init_generators(self) -> None:
        self.ob_generator = LaneVehicleGenerator(
            self.world, self.inter_obj, ["lane_count"], in_only=True, average=None
        )
        self.phase_generator = IntersectionPhaseGenerator(
            self.world, self.inter_obj, ["phase"], targets=["cur_phase"], negative=False
        )
        self.reward_generator = LaneVehicleGenerator(
            self.world,
            self.inter_obj,
            ["lane_count"],
            in_only=True,
            average="all",
            negative=True,
        )
        self.queue = LaneVehicleGenerator(
            self.world,
            self.inter_obj,
            ["lane_waiting_count"],
            in_only=True,
            negative=False,
        )
        self.delay = LaneVehicleGenerator(
            self.world, self.inter_obj, ["lane_delay"], in_only=True, negative=False
        )

    def __repr__(self) -> str:
        backend = self.param.get("backend", "api")
        return (
            f"TrafficR1Agent(rank={self.rank}, id={self.inter_obj.id}, "
            f"backend={backend}, llm={self._llm_enabled}, "
            f"phase_map={self._signal_to_phase})"
        )

    def reset(self) -> None:
        inter_id = self.world.intersection_ids[self.rank]
        self.inter_obj = self.world.id2intersection[inter_id]
        self._init_generators()
        self._lane_meta = self._build_lane_meta()
        self._signal_to_phase = self._resolve_phase_map(self.param)
        self._llm_enabled = len(self.inter_obj.phases) >= 4 and bool(
            self._signal_to_phase
        )
        self.last_action = 0
        self.last_raw_response = None
        self.last_signal = None

    # ---- observation / reward / phase (LibSignal interface) ----

    def get_ob(self):
        x_obs = [self.ob_generator.generate()]
        return np.array(x_obs, dtype=np.float32)

    def get_reward(self):
        rewards = [self.reward_generator.generate()]
        return np.squeeze(np.array(rewards)) * 12

    def get_phase(self):
        phase = [self.phase_generator.generate()]
        return (np.concatenate(phase)).astype(np.int8)

    def get_queue(self):
        queue = [self.queue.generate()]
        return np.sum(np.squeeze(np.array(queue)))

    def get_delay(self):
        delay = [self.delay.generate()]
        return np.sum(np.squeeze(np.array(delay)))

    # ---- action ----

    def get_action(self, ob, phase, test=True):
        # Fringe / single-phase TLS: no LLM.
        if not self._llm_enabled:
            self.last_action = 0
            self.last_signal = None
            return 0

        if self._backend is None:
            self._backend = _get_shared_backend(self.param)

        signal_stats = self._collect_signal_stats()
        messages = build_messages(signal_stats, n_segments=self.n_segments)

        last_err: Optional[Exception] = None
        for attempt in range(1, self.parse_retries + 1):
            raw = self._backend.complete(messages)
            self.last_raw_response = raw
            try:
                signal = parse_signal(raw, allowed=SIGNAL_ORDER)
            except TrafficR1ParseError as e:
                last_err = e
                print(
                    f"[Traffic-R1] parse failure at {self.inter_obj.id} "
                    f"attempt {attempt}/{self.parse_retries}: {e}"
                )
                continue
            if signal not in self._signal_to_phase:
                last_err = TrafficR1ParseError(
                    f"Signal {signal} has no mapped green index on {self.inter_obj.id}"
                )
                print(
                    f"[Traffic-R1] mapping failure at {self.inter_obj.id} "
                    f"attempt {attempt}/{self.parse_retries}: {last_err}"
                )
                continue
            action = int(self._signal_to_phase[signal])
            self.last_action = action
            self.last_signal = signal
            return action

        raise RuntimeError(
            f"Traffic-R1 hard failure at intersection {self.inter_obj.id}: "
            f"could not parse a valid signal after {self.parse_retries} attempts. "
            f"Last error: {last_err}. Last response tail: "
            f"{(self.last_raw_response or '')[-500:]!r}"
        )

    # ---- lane / phase geometry ----

    def _build_lane_meta(
        self,
    ) -> Dict[str, Tuple[str, str]]:
        """
        Map incoming lane id -> (approach_from, movement) with movement in R/T/L.

        Approach is where traffic comes from. For 3-lane approaches we assign
        right/through/left by ascending SUMO lane index (grid4x4 convention).
        """
        meta: Dict[str, Tuple[str, str]] = {}
        inter = self.inter_obj
        for road, direction, is_out in zip(inter.roads, inter.directions, inter.outs):
            if is_out:
                continue
            approach = _compass_from_atan2(direction)
            lanes = sorted(
                inter.road_lane_mapping[road],
                key=lambda x: int(str(x).rsplit("_", 1)[-1]),
            )
            if len(lanes) >= 3:
                labels = ["R", "T", "L"] + ["L"] * (len(lanes) - 3)
            elif len(lanes) == 2:
                labels = ["T", "L"]
            elif len(lanes) == 1:
                labels = ["T"]
            else:
                continue
            for lane, lab in zip(lanes, labels):
                meta[lane] = (approach, lab)
        return meta

    def _resolve_phase_map(self, param: Dict[str, Any]) -> Dict[str, int]:
        """Map Traffic-R1 signal names to LibSignal green-phase indices."""
        explicit = param.get("phase_name_to_index")
        if explicit:
            return {str(k).upper(): int(v) for k, v in explicit.items()}

        n = len(self.inter_obj.phases)
        if n == 1:
            return {}
        if n == 4:
            # Assume CityFlow-style ordering if an intersection already has 4 greens.
            # Prefer auto-detect below when lane meta is rich enough.
            auto = self._autodetect_phase_map()
            if auto:
                return auto
            return {name: i for i, name in enumerate(SIGNAL_ORDER)}

        auto = self._autodetect_phase_map()
        if auto:
            return auto

        # Documented Grid4x4 / NEMA ordering fallback used in this repo.
        fallback = {"NTST": 0, "NLSL": 1, "ETWT": 4, "ELWL": 5}
        if n >= 6 and all(idx < n for idx in fallback.values()):
            print(
                f"[Traffic-R1] Warning: using NEMA fallback phase map on "
                f"{self.inter_obj.id}: {fallback}"
            )
            return fallback
        raise RuntimeError(
            f"Traffic-R1 could not map 4 signal names onto {n} green phases "
            f"at intersection {self.inter_obj.id}"
        )

    def _autodetect_phase_map(self) -> Dict[str, int]:
        """
        Find green indices whose non-right start lanes match each 4-phase pair.
        """
        if not self._lane_meta:
            return {}
        phase_sets: List[frozenset] = []
        for phase_id in range(len(self.inter_obj.phases)):
            keys = []
            for lane in self.inter_obj.phase_available_startlanes[phase_id]:
                if lane not in self._lane_meta:
                    continue
                approach, mov = self._lane_meta[lane]
                if mov == "R":
                    continue
                keys.append((approach, mov))
            phase_sets.append(frozenset(keys))

        mapping: Dict[str, int] = {}
        for signal, pair in SIGNAL_LANE_PAIRS.items():
            target = frozenset(pair)
            matches = [i for i, s in enumerate(phase_sets) if s == target]
            if len(matches) == 1:
                mapping[signal] = matches[0]
            elif len(matches) > 1:
                # Prefer the phase that does not also include extras (exact match already).
                mapping[signal] = matches[0]
        if len(mapping) == 4:
            return mapping
        return {}

    def _collect_signal_stats(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        stats = empty_signal_stats(n_segments=self.n_segments)
        fo = self.inter_obj.full_observation
        if not fo:
            return stats

        # Aggregate early_queued + approaching segments per (approach, movement).
        lane_counts: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for lane, (approach, mov) in self._lane_meta.items():
            if mov == "R":
                continue
            lane_obs = fo.get(lane)
            if not lane_obs:
                continue
            length = float(self.world.eng.lane.getLength(lane))
            early = 0
            segments = [0] * self.n_segments
            for v in lane_obs.get("vehicles", []):
                speed = float(v.get("speed", 0.0))
                pos = float(v.get("position", 0.0))
                dist_to_stop = max(0.0, length - pos)
                if dist_to_stop > self.obs_distance:
                    continue
                if speed < self.v_stop:
                    early += 1
                    continue
                # Approaching: bin into segments (1 = closest to intersection).
                if length <= 0:
                    continue
                # Relative position from stop line in [0, 1].
                rel = min(1.0, dist_to_stop / length)
                seg_idx = min(self.n_segments - 1, int(rel * self.n_segments))
                segments[seg_idx] += 1
            lane_counts[(approach, mov)] = {
                "early_queued": early,
                "segments": segments,
            }

        for signal, pair in SIGNAL_LANE_PAIRS.items():
            for approach, mov in pair:
                side = APPROACH_WORD[approach]
                data = lane_counts.get(
                    (approach, mov),
                    {"early_queued": 0, "segments": [0] * self.n_segments},
                )
                stats[signal][side] = {
                    "early_queued": int(data["early_queued"]),
                    "segments": [int(x) for x in data["segments"]],
                }
        return stats
