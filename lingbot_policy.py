"""
LingBot-VLA 2.0 LoRA policy for UniBot V1 Challenge.

Implements the competition policy interface:
  - metadata        → dict describing obs/action contract
  - get_action(obs) → action chunk dict
  - reset()         → clear per-episode state

This is a thin adapter between the competition interface and the project's
deploy server (deploy/lingbot_vla_v2_policy.py:LingbotVLAv2Server), which
reproduces the training-time feature transforms exactly (robot-config key
mapping, meanstd normalization, subtract_state delta handling, camera
preprocessing).

Model: LingBot-VLA 2.0 (6.3B) — Qwen3-VL-4B-Instruct backbone + Qwen2 action
expert, LoRA rank=64 (q/v/o proj) fine-tuned on the UniBot G1_Dex1 dataset,
merged into the base weights as full-model safetensors
(scripts/export_unibot_lora_merged.py).

Control space: "joint" — the checkpoint predicts arm/effector/waist/base in
the canonical space; waist joints are not part of the competition action
spec and are dropped, pivot[3:6] (constant in the data) are re-filled with
the dataset constants (0, 0.149, 0).

Requirements:
  - Single RTX 4090 (24GB VRAM)
  - lingbot-vla-v2 repo on PYTHONPATH (LINGBOT_ROOT)
  - Merged safetensors export at MODEL_PATH
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
LINGBOT_ROOT = os.environ.get("LINGBOT_ROOT", "/mnt/ssd/lingbot-vla-v2")
if LINGBOT_ROOT not in sys.path:
    sys.path.insert(0, LINGBOT_ROOT)

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    f"{LINGBOT_ROOT}/output/unibot_lora_full/checkpoints/global_step_step=10000/model",
)
NORM_STATS_PATH = os.environ.get(
    "NORM_STATS_PATH", f"{LINGBOT_ROOT}/assets/norm_stats/unibot_full.json"
)
ROBOT_NAME = os.environ.get("ROBOT_NAME", "unibot/g1_dex1_full")

# Competition observation key → dataset (origin) feature key, matching
# configs/robot_configs/unibot/g1_dex1_full.yaml origin_keys.
CAMERA_KEY_MAP = {
    "observation.images.cam_left_high": "observation.images.head_stereo_left",
    "observation.images.cam_right_high": "observation.images.head_stereo_right",
    "observation.images.cam_left_wrist": "observation.images.wrist_left",
    "observation.images.cam_right_wrist": "observation.images.wrist_right",
}

# lower_body (15): [0:12] legs, [12:15] waist_yaw/roll/pitch → waist_state_joint
LOWER_BODY_WAIST_SLICE = slice(12, 15)

# action.pivot slots reconstructed from the canonical space:
#   base.position[0:3]  → pivot[0:3] (vx, vy, angle_z)
#   waist.position[3]   → pivot[6]   (body height)
#   pivot[3:6] carry no learnable signal (constants 0, ~0.149, 0 in the data)
#   and were dropped during training — re-filled with those constants here.
PIVOT_DIM = 7
PIVOT_CONSTANT_FILL = (0.0, 0.149, 0.0)  # pivot[3], pivot[4], pivot[5]


class LingBotPolicy:
    """UniBot-V1 policy backed by a LoRA-fine-tuned LingBot-VLA 2.0 model."""

    OBS_CHUNK_SIZE = 1
    ACTION_CHUNK_SIZE = 50
    DATA_KEYS = (
        "observation.language",
        "observation.images.cam_left_high",
        "observation.images.cam_right_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
        "observation.state.left_arm",
        "observation.state.right_arm",
        "observation.state.left_gripper",
        "observation.state.right_gripper",
        "observation.state.lower_body",
    )

    def __init__(self):
        self._token = os.environ.get("UNIBOT_SUBMISSION_TOKEN")
        if self._token is None:
            raise ValueError("UNIBOT_SUBMISSION_TOKEN is required")
        self._control_space = os.environ.get("UNIBOT_CONTROL_SPACE", "joint")
        if self._control_space != "joint":
            raise ValueError(
                "This checkpoint was trained on joint-space actions only; "
                f"UNIBOT_CONTROL_SPACE must be 'joint', got {self._control_space!r}"
            )

        self._step = 0

        # deploy.lingbot_vla_v2_policy resolves the robot config as
        # configs/robot_configs/<robo_name>.yaml relative to the cwd.
        os.chdir(LINGBOT_ROOT)

        from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server

        log.info("Loading LingBot-VLA 2.0 (merged LoRA export) from %s …", MODEL_PATH)
        self._server = LingbotVLAv2Server(
            path_to_pi_model=MODEL_PATH,
            robot_norm_path=NORM_STATS_PATH,
            use_length=self.ACTION_CHUNK_SIZE,
            chunk_ret=True,  # return the full 50-step action chunk per call
            use_bf16=True,
        )
        self._server.reset(ROBOT_NAME)
        log.info("LingBotPolicy ready")

    # ═══════════════════════════════════════════════════════════════════════
    # Competition interface
    # ═══════════════════════════════════════════════════════════════════════

    @property
    def metadata(self) -> dict:
        return {
            "control_space": self._control_space,
            "data_keys": list(self.DATA_KEYS),
            "obs_chunk_size": self.OBS_CHUNK_SIZE,
            "action_chunk_size": self.ACTION_CHUNK_SIZE,
            "token": self._token,
        }

    def get_action(self, obs: dict) -> dict:
        """Compute one action chunk from the competition observation dict."""
        import time

        t_start = time.perf_counter()

        ds_obs = self._obs_to_dataset(obs)
        out = self._server.infer(ds_obs)  # {action.*: (50, dim) float32}
        action = self._dataset_to_action(out)

        elapsed = time.perf_counter() - t_start
        self._step += 1
        log.info("[step %d] inference %.2fs", self._step, elapsed)
        return action

    def reset(self):
        self._step = 0
        self._server.reset(ROBOT_NAME)
        return {"ok": True}

    # ═══════════════════════════════════════════════════════════════════════
    # Observation conversion: competition keys → dataset origin keys
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _frame(value):
        """(T, ...) → last frame (...); pass through non-batched values."""
        arr = np.asarray(value)
        if arr.ndim > 1 and arr.shape[0] == LingBotPolicy.OBS_CHUNK_SIZE:
            arr = arr[-1]
        return arr

    def _obs_to_dataset(self, obs: dict) -> dict:
        ds_obs = {}
        for comp_key, ds_key in CAMERA_KEY_MAP.items():
            ds_obs[ds_key] = np.ascontiguousarray(self._frame(obs[comp_key]))

        ds_obs["observation.state.left_arm"] = self._frame(
            obs["observation.state.left_arm"]
        ).astype(np.float32)
        ds_obs["observation.state.right_arm"] = self._frame(
            obs["observation.state.right_arm"]
        ).astype(np.float32)
        ds_obs["observation.state.left_gripper"] = self._frame(
            obs["observation.state.left_gripper"]
        ).astype(np.float32)
        ds_obs["observation.state.right_gripper"] = self._frame(
            obs["observation.state.right_gripper"]
        ).astype(np.float32)
        ds_obs["observation.state.waist_state_joint"] = self._frame(
            obs["observation.state.lower_body"]
        ).astype(np.float32)[LOWER_BODY_WAIST_SLICE]

        language = obs["observation.language"]
        if isinstance(language, np.ndarray):
            language = str(language)
        ds_obs["task"] = language

        return ds_obs

    # ═══════════════════════════════════════════════════════════════════════
    # Action conversion: dataset origin actions → competition action dict
    # ═══════════════════════════════════════════════════════════════════════

    def _dataset_to_action(self, out: dict) -> dict:
        T = self.ACTION_CHUNK_SIZE

        pivot4 = np.asarray(out["action.pivot"], dtype=np.float32)  # (T, 4)
        pivot = np.zeros((T, PIVOT_DIM), dtype=np.float32)
        pivot[:, 0:3] = pivot4[:, 0:3]  # vx, vy, angle_z (base command)
        pivot[:, 3:6] = PIVOT_CONSTANT_FILL  # fixed fields dropped in training
        pivot[:, 6] = pivot4[:, 3]      # body height (4th waist slot)

        return {
            "meta.token": self._token,
            "action.left_arm": np.asarray(out["action.left_arm"], dtype=np.float32),
            "action.right_arm": np.asarray(out["action.right_arm"], dtype=np.float32),
            "action.left_gripper": np.asarray(out["action.left_gripper"], dtype=np.float32),
            "action.right_gripper": np.asarray(out["action.right_gripper"], dtype=np.float32),
            "action.pivot": pivot,
        }
