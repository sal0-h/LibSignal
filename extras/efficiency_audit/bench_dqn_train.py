#!/usr/bin/env python3
"""
Microbenchmark: DQN train() and get_action() on CPU for sumo1x1-scale MLP.

Usage (repo root, venv activated):
  python extras/efficiency_audit/bench_dqn_train.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_PATH = Path(__file__).resolve().parent / "dqn_train_bench.json"

# Typical sumo1x1 DQN dims (lane_count + one-hot phase); cologne1-backed sumo1x1 uses
# more lanes (~52) but the MLP is identical — we benchmark the canonical small case.
LANE_OB_LENGTH = 12
N_ACTIONS = 8
INPUT_DIM = LANE_OB_LENGTH + N_ACTIONS
BATCH_SIZE = 64
GAMMA = 0.95
LEARNING_RATE = 0.001
GRAD_CLIP = 5.0
WARMUP_ITERS = 50
BENCH_ITERS = 200
DECISIONS_PER_EPISODE = 360
OBSERVED_ENV_EPISODE_S = 1.0  # ~1s env episode on sumo1x1 (mp_baseline ~0.93–1.9s wall)


class DQNNet(nn.Module):
    """Same architecture as agent/dqn.py DQNNet."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.dense_1 = nn.Linear(input_dim, 20)
        self.dense_2 = nn.Linear(20, 20)
        self.dense_3 = nn.Linear(20, output_dim)

    def _forward(self, x):
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = self.dense_3(x)
        return x

    def forward(self, x, train=True):
        if train:
            return self._forward(x)
        with torch.no_grad():
            return self._forward(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_flops_per_forward(batch: int, input_dim: int, hidden: int, output_dim: int) -> int:
    """Approximate GEMM FLOPs (2 mul-add per weight) for one forward pass."""
    flops = 0
    flops += 2 * batch * input_dim * hidden  # dense_1
    flops += 2 * batch * hidden * hidden   # dense_2
    flops += 2 * batch * hidden * output_dim  # dense_3
    return flops


def train_step(
    model: DQNNet,
    target_model: DQNNet,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    b_t: torch.Tensor,
    b_tp: torch.Tensor,
    rewards: torch.Tensor,
    actions: torch.Tensor,
    gamma: float,
    grad_clip: float,
) -> None:
    """Mirror agent/dqn.py DQNAgent.train() hot path."""
    out = target_model(b_tp, train=False)
    target = rewards + gamma * torch.max(out, dim=1)[0]
    target_f = model(b_t, train=False)
    for i, action in enumerate(actions):
        target_f[i][action] = target[i]
    loss = criterion(model(b_t, train=True), target_f)
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()


def get_action_forward(model: DQNNet, feature: torch.Tensor) -> np.ndarray:
    """Mirror get_action inference path (batch=1, train=False)."""
    actions = model(feature, train=False)
    actions = actions.cpu().detach().numpy()
    return np.argmax(actions, axis=1)


def percentile_ms(samples_ms: list[float], p: float) -> float:
    return float(np.percentile(samples_ms, p))


def try_probe_sumo1x1_dims() -> dict | None:
    """Best-effort probe of live sumo1x1 ob_length if SUMO/registry available."""
    try:
        from common.registry import Registry
        from common import interface
        from common.utils import build_config
        import argparse

        pargs = argparse.Namespace(
            thread_num=1,
            ngpu="-1",
            prefix="bench_probe",
            seed=42,
            debug=False,
            interface="libsumo",
            delay_type="apx",
            task="tsc",
            agent="dqn",
            world="sumo",
            network="sumo1x1",
            dataset="onfly",
        )
        config, _ = build_config(pargs)
        interface.Command_Setting_Interface(config)
        interface.World_param_Interface(config)
        interface.ModelAgent_param_Interface(config)

        from world.world_sumo import World
        from generator import LaneVehicleGenerator

        world = World(
            Registry.mapping["world_mapping"]["setting"].param["combined_file"],
            42,
            interface=Registry.mapping["command_mapping"]["setting"].param.get("interface", "libsumo"),
        )
        inter_id = world.intersection_ids[0]
        inter = world.id2intersection[inter_id]
        ob_gen = LaneVehicleGenerator(world, inter, ["lane_count"], in_only=True, average=None)
        n_phases = len(inter.phases)
        world.eng.close()
        return {
            "lane_ob_length": int(ob_gen.ob_length),
            "n_phases": int(n_phases),
            "input_dim_one_hot": int(ob_gen.ob_length + n_phases),
            "source": "live_sumo1x1_probe",
        }
    except Exception as exc:  # noqa: BLE001 — probe is optional
        return {"error": str(exc), "source": "probe_failed"}


def main() -> dict:
    device = torch.device("cpu")
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    model = DQNNet(INPUT_DIM, N_ACTIONS).to(device)
    target_model = DQNNet(INPUT_DIM, N_ACTIONS).to(device)
    target_model.load_state_dict(model.state_dict())
    criterion = nn.MSELoss(reduction="mean")
    optimizer = optim.RMSprop(
        model.parameters(), lr=LEARNING_RATE, alpha=0.9, centered=False, eps=1e-7
    )

    n_params = count_parameters(model)
    hidden = 20
    flops_fwd_b1 = estimate_flops_per_forward(1, INPUT_DIM, hidden, N_ACTIONS)
    flops_fwd_b64 = estimate_flops_per_forward(BATCH_SIZE, INPUT_DIM, hidden, N_ACTIONS)
    # train(): target forward + 2x policy forward + backward ~3x forward FLOPs
    flops_train_approx = flops_fwd_b64 * 3 + flops_fwd_b64 * 3  # fwd + bwd rough

    rng = np.random.default_rng(42)
    b_t = torch.tensor(rng.random((BATCH_SIZE, INPUT_DIM), dtype=np.float32), device=device)
    b_tp = torch.tensor(rng.random((BATCH_SIZE, INPUT_DIM), dtype=np.float32), device=device)
    rewards = torch.tensor(rng.random(BATCH_SIZE, dtype=np.float32), device=device)
    actions = torch.tensor(rng.integers(0, N_ACTIONS, size=BATCH_SIZE), device=device, dtype=torch.long)
    feature_b1 = torch.tensor(rng.random((1, INPUT_DIM), dtype=np.float32), device=device)

    # Warmup
    for _ in range(WARMUP_ITERS):
        train_step(
            model, target_model, optimizer, criterion,
            b_t, b_tp, rewards, actions, GAMMA, GRAD_CLIP,
        )
        _ = get_action_forward(model, feature_b1)

    train_ms: list[float] = []
    for _ in range(BENCH_ITERS):
        t0 = time.perf_counter()
        train_step(
            model, target_model, optimizer, criterion,
            b_t, b_tp, rewards, actions, GAMMA, GRAD_CLIP,
        )
        train_ms.append((time.perf_counter() - t0) * 1000.0)

    action_ms: list[float] = []
    for _ in range(BENCH_ITERS):
        t0 = time.perf_counter()
        _ = get_action_forward(model, feature_b1)
        action_ms.append((time.perf_counter() - t0) * 1000.0)

    train_mean = float(np.mean(train_ms))
    train_median = float(np.median(train_ms))
    action_mean = float(np.mean(action_ms))
    action_median = float(np.median(action_ms))

    rl_train_per_episode_s = (train_mean / 1000.0) * DECISIONS_PER_EPISODE
    rl_action_per_episode_s = (action_mean / 1000.0) * DECISIONS_PER_EPISODE
    rl_total_per_episode_s = rl_train_per_episode_s + rl_action_per_episode_s
    rl_fraction_of_episode = rl_total_per_episode_s / OBSERVED_ENV_EPISODE_S

    # GPU transfer overhead estimate: PCIe round-trip for tiny tensors
    # batch64 state ~20 floats * 4B * 2 tensors ~ 1.5KB; batch1 ~80B
    bytes_per_train_batch = BATCH_SIZE * INPUT_DIM * 4 * 2  # b_t + b_tp host tensors
    bytes_per_action = INPUT_DIM * 4
    pcie_gbps_typical = 16.0  # PCIe 3.0 x16 one-way theoretical GB/s
    pcie_train_xfer_ms = (bytes_per_train_batch / (pcie_gbps_typical * 1e9)) * 1000.0 * 2
    pcie_action_xfer_ms = (bytes_per_action / (pcie_gbps_typical * 1e9)) * 1000.0 * 2

    # Kernel launch / sync floor on GPU for tiny ops (rule-of-thumb)
    gpu_launch_floor_ms = 0.05
    gpu_train_est_ms = max(gpu_launch_floor_ms, flops_train_approx / (1e9 * 0.5))  # 0.5 TFLOPS util
    gpu_action_est_ms = max(gpu_launch_floor_ms, (flops_fwd_b1 * 2) / (1e9 * 0.5))

    probe = try_probe_sumo1x1_dims()

    results = {
        "torch_version": torch.__version__,
        "device": str(device),
        "cpu_threads": torch.get_num_threads(),
        "architecture": {
            "class": "DQNNet",
            "layers": [
                {"name": "dense_1", "in": INPUT_DIM, "out": hidden, "activation": "relu"},
                {"name": "dense_2", "in": hidden, "out": hidden, "activation": "relu"},
                {"name": "dense_3", "in": hidden, "out": N_ACTIONS, "activation": "linear"},
            ],
            "hidden_size": hidden,
            "n_parameters": n_params,
            "source_file": "agent/dqn.py::_build_model / DQNNet",
        },
        "benchmark_config": {
            "lane_ob_length": LANE_OB_LENGTH,
            "n_actions_phases": N_ACTIONS,
            "input_dim": INPUT_DIM,
            "batch_size": BATCH_SIZE,
            "warmup_iters": WARMUP_ITERS,
            "bench_iters": BENCH_ITERS,
            "gamma": GAMMA,
            "learning_rate": LEARNING_RATE,
            "grad_clip": GRAD_CLIP,
        },
        "sumo1x1_probe": probe,
        "flops_estimate": {
            "forward_batch1": flops_fwd_b1,
            "forward_batch64": flops_fwd_b64,
            "train_step_approx": flops_train_approx,
            "note": "GEMM-only; ReLU/backward counted roughly as 3x forward in train_step_approx",
        },
        "timing_ms": {
            "train_step": {
                "mean": round(train_mean, 4),
                "median": round(train_median, 4),
                "std": round(float(np.std(train_ms)), 4),
                "p5": round(percentile_ms(train_ms, 5), 4),
                "p95": round(percentile_ms(train_ms, 95), 4),
                "min": round(float(np.min(train_ms)), 4),
                "max": round(float(np.max(train_ms)), 4),
                "n": BENCH_ITERS,
            },
            "get_action_forward_batch1": {
                "mean": round(action_mean, 4),
                "median": round(action_median, 4),
                "std": round(float(np.std(action_ms)), 4),
                "p5": round(percentile_ms(action_ms, 5), 4),
                "p95": round(percentile_ms(action_ms, 95), 4),
                "min": round(float(np.min(action_ms)), 4),
                "max": round(float(np.max(action_ms)), 4),
                "n": BENCH_ITERS,
            },
        },
        "episode_estimate": {
            "decisions_per_episode": DECISIONS_PER_EPISODE,
            "assumption": "train() every decision after learning_start (steady-state)",
            "observed_env_episode_s": OBSERVED_ENV_EPISODE_S,
            "rl_train_s_per_episode": round(rl_train_per_episode_s, 4),
            "rl_get_action_s_per_episode": round(rl_action_per_episode_s, 4),
            "rl_total_s_per_episode": round(rl_total_per_episode_s, 4),
            "rl_fraction_of_episode_wall_time": round(rl_fraction_of_episode, 4),
            "env_dominates": rl_fraction_of_episode < 0.5,
        },
        "gpu_analysis": {
            "model_size_bytes": n_params * 4,
            "pcie_transfer_est_ms": {
                "per_train_step_batch64_roundtrip": round(pcie_train_xfer_ms, 6),
                "per_get_action_batch1_roundtrip": round(pcie_action_xfer_ms, 6),
            },
            "theoretical_gpu_compute_floor_ms": {
                "train_step_at_0p5_tflops": round(gpu_train_est_ms, 4),
                "get_action_at_0p5_tflops": round(gpu_action_est_ms, 4),
            },
            "cpu_train_ms_mean": round(train_mean, 4),
            "verdict": (
                "GPU unlikely to help wall-clock training on sumo1x1: model is 1008 params "
                f"({flops_train_approx} FLOPs/train), CPU completes train in ~{train_mean:.2f}ms while "
                f"env episode is ~{OBSERVED_ENV_EPISODE_S}s. PCIe sync/launch overhead (~{gpu_launch_floor_ms}ms) "
                "exceeds compute savings for batch=64; RL is env-bound not compute-bound."
            ),
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=" * 60)
    print("DQN train microbenchmark (CPU)")
    print("=" * 60)
    print(f"Architecture: {INPUT_DIM} -> 20 -> 20 -> {N_ACTIONS}  ({n_params} params)")
    print(f"train() step:  mean={train_mean:.3f} ms  median={train_median:.3f} ms")
    print(f"get_action:    mean={action_mean:.3f} ms  median={action_median:.3f} ms")
    print(f"Episode RL est: train={rl_train_per_episode_s:.3f}s + action={rl_action_per_episode_s:.3f}s "
          f"= {rl_total_per_episode_s:.3f}s vs env ~{OBSERVED_ENV_EPISODE_S}s "
          f"({100*rl_fraction_of_episode:.1f}% of episode)")
    print(f"GPU verdict: {results['gpu_analysis']['verdict']}")
    print(f"Wrote {OUT_PATH}")
    return results


if __name__ == "__main__":
    main()
