"""
LingBot-VLA 2.0 LoRA policy for UniBot V1 Challenge.

Implements the competition policy interface (§1 of the submission spec):
  - metadata  → dict describing obs/action contract
  - get_action(obs) → action chunk dict
  - reset()   → clear per-episode state

Architecture:
  LingBot-VLA 2.0 (6.3B): Qwen3-VL-4B-Instruct backbone + Qwen2 action expert
  with MoE layers. LoRA rank=64 on q_proj/v_proj/o_proj attention projections.
  Fine-tuned on UniBot ArrangePlates task (right arm + gripper, single camera).

Control space: "joint" — predicts target joint positions for right arm (7 DOF)
  and right gripper (1 DOF). Left arm, left gripper, and pivot are held at the
  most recent observed state.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Paths — adjust LINGBOT_ROOT to point at the lingbot-vla-v2 clone
# ──────────────────────────────────────────────────────────────────────────────
LINGBOT_ROOT = os.environ.get("LINGBOT_ROOT", "/mnt/ssd/lingbot-vla-v2")
sys.path.insert(0, LINGBOT_ROOT)

# Model & checkpoint paths
PRETRAINED_MODEL_PATH = os.environ.get(
    "PRETRAINED_MODEL_PATH",
    "/mnt/ssd/lingbot-vla-v2/pretrained_models/lingbot-vla-v2-6b",
)
TOKENIZER_PATH = os.environ.get(
    "TOKENIZER_PATH",
    "/mnt/ssd/lingbot-vla-v2/pretrained_models/Qwen3-VL-4B-Instruct",
)
LORA_CHECKPOINT_PATH = os.environ.get(
    "LORA_CHECKPOINT_PATH",
    "/home/zireael/lingbot_output/unibot_lora/checkpoints/global_step_4000/model",
)
NORM_STATS_PATH = os.environ.get(
    "NORM_STATS_PATH",
    "/home/zireael/lingbot_output/norm_stats/g1_dex1.json",
)

# ──────────────────────────────────────────────────────────────────────────────
# Canonical space dimensions (from training config)
# ──────────────────────────────────────────────────────────────────────────────
CANONICAL_JOINTS = {
    "arm.position":      14,   # canonical arm dims (7 real for right arm + 7 padding)
    "effector.position":  2,   # canonical effector dims (1 real for right gripper + 1 padding)
}
CANONICAL_ORDER = ["arm.position", "effector.position"]
MAX_ACTION_DIM = 55
MAX_STATE_DIM = 55
N_ACTION_STEPS = 50  # chunk_size from training config
NUM_DENOISE_STEPS = 10  # flow-matching sampling steps


class LingBotPolicy:
    """UniBot-V1 policy backed by a LoRA-fine-tuned LingBot-VLA 2.0 model.

    The model predicts right-arm joint positions and right-gripper opening.
    All other action dimensions (left arm, left gripper, pivot) are held at
    the most recently observed state.
    """

    # ── Competition metadata ──────────────────────────────────────────────
    OBS_CHUNK_SIZE = 1       # single-frame observation
    ACTION_CHUNK_SIZE = 50   # matches model's n_action_steps
    DATA_KEYS = (
        "observation.language",
        "observation.images.cam_left_high",
        "observation.state.right_arm",
        "observation.state.right_gripper",
    )

    def __init__(self):
        self._token = os.environ.get("UNIBOT_SUBMISSION_TOKEN")
        if self._token is None:
            raise ValueError("UNIBOT_SUBMISSION_TOKEN is required")
        self._control_space = os.environ.get("UNIBOT_CONTROL_SPACE", "joint")
        if self._control_space not in ("joint", "ee"):
            raise ValueError(
                f"UNIBOT_CONTROL_SPACE must be 'joint' or 'ee', "
                f"got {self._control_space!r}"
            )

        self._step = 0
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        log.info("Loading LingBot-VLA 2.0 with LoRA …")
        self._load_model()
        self._load_norm_stats()
        log.info("LingBotPolicy ready (%.1fM trainable / %.1fM total params)",
                 sum(p.numel() for p in self._model.parameters() if p.requires_grad) / 1e6,
                 sum(p.numel() for p in self._model.parameters()) / 1e6)

    # ═══════════════════════════════════════════════════════════════════════
    # Competition interface
    # ═══════════════════════════════════════════════════════════════════════

    @property
    def metadata(self) -> dict:
        return {
            "control_space":      self._control_space,
            "data_keys":          list(self.DATA_KEYS),
            "obs_chunk_size":     self.OBS_CHUNK_SIZE,
            "action_chunk_size":  self.ACTION_CHUNK_SIZE,
            "token":              self._token,
        }

    def get_action(self, obs: dict) -> dict:
        """Compute one action chunk from the competition observation dict."""
        t_start = time.perf_counter()

        # 1. Build model inputs from competition-format observation
        model_inputs = self._obs_to_model(obs)

        # 2. Run flow-matching inference → canonical action (50, 55)
        with torch.inference_mode():
            canonical_actions = self._model.sample_actions(
                images=model_inputs["images"],
                img_masks=model_inputs["img_masks"],
                lang_tokens=model_inputs["lang_tokens"],
                lang_masks=model_inputs["lang_masks"],
                state=model_inputs["state"],
                image_grid_thw=model_inputs["image_grid_thw"],
            )  # (1, n_action_steps, max_action_dim) = (1, 50, 55)

        # 3. Convert canonical action → competition action dict
        action = self._model_to_action(
            canonical_actions,
            model_inputs,
        )

        elapsed = time.perf_counter() - t_start
        self._step += 1
        log.info("[step %d] inference %.2fs", self._step, elapsed)
        return action

    def reset(self):
        """Clear per-episode state."""
        self._step = 0
        return {"ok": True}

    # ═══════════════════════════════════════════════════════════════════════
    # Model loading
    # ═══════════════════════════════════════════════════════════════════════

    def _load_model(self):
        """Build the LingBot-VLA model, apply LoRA, and load the DCP checkpoint."""
        from lingbotvla.models import build_foundation_model, build_processor
        from lingbotvla.utils.lora_utils import add_lora_to_model, freeze_parameters
        from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import (
            FlowMatchingV2,
            QwenvlWithExpertV2Config,
        )

        # --- Build FlowMatchingV2 config ---
        fm_config = self._build_flow_matching_config()

        # --- Build foundation model ---
        log.info("Building foundation model (this loads ~12GB of weights) …")
        model = build_foundation_model(
            config_path=None,
            config_cls=fm_config,
            weights_path=PRETRAINED_MODEL_PATH,
            torch_dtype="bfloat16",
            init_device="cpu",
            force_use_huggingface=False,
            config_kwargs={},
            moe_implementation="eager",
        )
        # build_foundation_model returns a FlowMatchingV2 instance when config_cls is set

        # --- Apply LoRA ---
        log.info("Applying LoRA structure (rank=64) …")
        lora_target_modules = "q_proj,v_proj,o_proj"
        lora_target_modules_support = set()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                lora_target_modules_support.add(name.split(".")[-1])

        add_lora_to_model(
            model,
            lora_rank=64,
            lora_alpha=32,
            lora_target_modules=lora_target_modules,
            init_lora_weights="kaiming",
            lora_target_modules_support=lora_target_modules_support,
        )

        # --- Load DCP checkpoint ---
        ckpt_path = Path(LORA_CHECKPOINT_PATH)
        if ckpt_path.exists():
            log.info("Loading LoRA checkpoint from %s …", ckpt_path)
            self._load_dcp_checkpoint(model, ckpt_path)
        else:
            log.warning("Checkpoint %s not found — using base model only", ckpt_path)

        # --- Move to GPU ---
        model = model.to(dtype=torch.bfloat16).cuda()
        model.eval()
        self._model = model

        # --- Load processor ---
        log.info("Loading Qwen3-VL processor …")
        self._processor = build_processor(
            config_path=TOKENIZER_PATH,
            config_cls=fm_config,
        )
        self._image_processor = self._processor.image_processor

    def _build_flow_matching_config(self):
        """Build the FlowMatchingV2 configuration matching training settings."""
        from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import (
            LingbotVLAV2Config,
        )

        return LingbotVLAV2Config(
            # Model paths
            model_path=PRETRAINED_MODEL_PATH,
            tokenizer_path=TOKENIZER_PATH,
            # Architecture
            moe_implementation="eager",
            vit_attn_implementation="sdpa",
            attention_implementation="eager",
            post_training=False,
            adanorm_time=True,
            # MoE
            use_moe=True,
            token_moe_layers=list(range(36)),
            token_num_experts=32,
            token_top_k=4,
            token_moe_intermediate_size=512,
            token_shared_intermediate_size=704,
            bias_update_speed=0,
            router_activation="sigmoid",
            routed_scaling_factor=4.0,
            use_shared_expert_gate=False,
            # Action space
            action_dim=MAX_ACTION_DIM,
            max_action_dim=MAX_ACTION_DIM,
            max_state_dim=MAX_STATE_DIM,
            chunk_size=N_ACTION_STEPS,
            num_steps=NUM_DENOISE_STEPS,
            # Training flags (inference settings)
            freeze_vision_encoder=True,
            train_expert_only=False,
            use_cache=True,
            enable_expert_vision=False,
            # Vision
            precompute_grid_thw=True,
            qwen3vl_use_vision_boundaries=True,
            # Language
            tokenizer_max_length=72,
            use_qwen3_chat_template=False,
            vlm_causal=True,
            # Action expert
            enable_fp32=False,  # bf16 for inference
            expert_hidden_size=768,
            expert_intermediate_size=2752,
            action_num_attention_heads=32,
            action_num_key_value_heads=8,
            action_head_dim=128,
            # No distillation
            align_params={},
        )

    def _load_dcp_checkpoint(self, model, ckpt_path: Path):
        """Load a DCP (Distributed Checkpoint) into the model.

        The checkpoint was saved from a DDP-wrapped model, so its keys are
        prefixed with ``state.model.model.``.  We build a state-dict alias
        that maps those checkpoint keys to the unwrapped model's tensors,
        then let ``torch.distributed.checkpoint.load`` do the rest.
        """
        from torch.distributed.checkpoint import FileSystemReader
        from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
        from torch.distributed.checkpoint.state_dict import set_model_state_dict

        # Try the modern DCP API first
        try:
            import torch.distributed.checkpoint as dcp
            reader = FileSystemReader(ckpt_path)
            metadata = reader.read_metadata()

            # Build alias dict: checkpoint key → model tensor
            model_sd = model.state_dict()
            alias: dict = {}
            prefix = "state.model.model."
            for ckpt_key in metadata.state_dict_metadata:
                if ckpt_key.startswith(prefix):
                    model_key = ckpt_key[len(prefix):]
                    if model_key in model_sd:
                        alias[ckpt_key] = model_sd[model_key]
                    # else: key exists in checkpoint but not in model — skip

            if len(alias) == 0:
                log.warning("No matching keys between checkpoint and model. "
                            "Trying alternative loading method.")
                self._load_dcp_fallback(model, ckpt_path)
                return

            log.info("Loading %d tensors from DCP checkpoint …", len(alias))
            dcp.load(
                state_dict=alias,
                checkpoint_id=ckpt_path,
                storage_reader=reader,
                planner=DefaultLoadPlanner(allow_partial_load=True),
            )
            log.info("DCP checkpoint loaded successfully.")
        except Exception as e:
            log.warning("DCP load failed: %s — trying fallback", e)
            self._load_dcp_fallback(model, ckpt_path)

    def _load_dcp_fallback(self, model, ckpt_path: Path):
        """Fallback: manually load each .distcp file and map tensors.

        Each .distcp file is a regular PyTorch saved dict with tensor
        name → tensor pairs.
        """
        distcp_files = sorted(ckpt_path.glob("__*_*.distcp"))
        if not distcp_files:
            raise FileNotFoundError(f"No .distcp files in {ckpt_path}")

        model_sd = model.state_dict()
        prefix = "state.model.model."
        loaded = 0

        for fpath in distcp_files:
            shard = torch.load(fpath, map_location="cpu", weights_only=False)
            for ckpt_key, tensor in shard.items():
                if ckpt_key.startswith(prefix):
                    model_key = ckpt_key[len(prefix):]
                    if model_key in model_sd:
                        dst = model_sd[model_key]
                        # Handle shape mismatches (DDP vs non-DDP can shift dims)
                        if dst.shape == tensor.shape:
                            dst.copy_(tensor)
                        else:
                            # Try to reshape/view
                            try:
                                dst.copy_(tensor.reshape(dst.shape))
                            except Exception:
                                log.debug("Shape mismatch for %s: %s vs %s — skipping",
                                         model_key, tuple(dst.shape), tuple(tensor.shape))
                        loaded += 1
            del shard

        log.info("Fallback loader: loaded %d tensors from %d shards",
                 loaded, len(distcp_files))

        if loaded == 0:
            raise RuntimeError(
                f"Failed to load any tensors from {ckpt_path}. "
                f"Check the checkpoint prefix."
            )

    def _load_norm_stats(self):
        """Load normalization statistics for the G1 right arm + gripper."""
        with open(NORM_STATS_PATH) as f:
            stats = json.load(f)
        self._norm_stats = stats["norm_stats"]

    # ═══════════════════════════════════════════════════════════════════════
    # Observation → model input
    # ═══════════════════════════════════════════════════════════════════════

    def _obs_to_model(self, obs: dict) -> dict:
        """Convert competition-format observation to model inputs.

        Competition format (§2 of the spec):
          - observation.images.cam_left_high  (T_o, 480, 640, 3) uint8
          - observation.state.right_arm       (T_o, 7) float32
          - observation.state.right_gripper   (T_o, 1) float32
          - observation.language              str

        Model expects [batch=1] inputs with canonical-space state and
        Qwen3-VL-processed images plus language tokens.
        """
        # --- Image ---
        # Take the last frame from the observation chunk
        raw_img = obs["observation.images.cam_left_high"]
        if raw_img.ndim > 3:
            raw_img = raw_img[-1]  # (T_o, H, W, 3) → (H, W, 3)

        # Qwen3-VL image processor expects PIL images or numpy arrays in [0,255]
        # The competition gives uint8 HWC. Convert to CHW for the processor.
        if isinstance(raw_img, np.ndarray):
            img_tensor = torch.from_numpy(raw_img).float()  # (H, W, 3) float
        else:
            img_tensor = raw_img.float()

        # Resize to 224x224 (training resolution)
        img_tensor = img_tensor.permute(2, 0, 1)  # (3, H, W)
        img_tensor = F.interpolate(
            img_tensor.unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)  # (3, 224, 224)
        img_tensor = img_tensor.clamp(0, 255).to(torch.uint8)

        # Run through Qwen3-VL image processor
        processed = self._image_processor(img_tensor)
        images = processed["pixel_values"].unsqueeze(0).cuda()  # (1, n_tokens, dim)
        image_grid_thw = processed.get("image_grid_thw")
        if image_grid_thw is not None:
            if image_grid_thw.ndim == 1:
                image_grid_thw = image_grid_thw.unsqueeze(0)
            image_grid_thw = image_grid_thw.cuda()

        img_masks = torch.ones(1, 1, dtype=torch.bool, device=self._device)

        # --- State ---
        right_arm = obs["observation.state.right_arm"]
        if right_arm.ndim > 1:
            right_arm = right_arm[-1]  # take last frame → (7,)
        right_gripper = obs["observation.state.right_gripper"]
        if right_gripper.ndim > 1:
            right_gripper = right_gripper[-1]  # take last frame → (1,)

        # Convert to tensors
        arm_t = torch.as_tensor(right_arm, dtype=torch.float32)
        grip_t = torch.as_tensor(right_gripper, dtype=torch.float32)

        # Normalize: bounds_99_woclip
        arm_norm = self._normalize(
            arm_t,
            self._norm_stats["observation.state.arm.position"],
            norm_type="bounds_99_woclip",
        )
        grip_norm = self._normalize(
            grip_t,
            self._norm_stats["observation.state.effector.position"],
            norm_type="bounds_99_woclip",
        )

        # Pad to canonical dims: arm 7→14, effector 1→2
        arm_padded = F.pad(arm_norm, (0, CANONICAL_JOINTS["arm.position"] - arm_norm.shape[-1]))
        grip_padded = F.pad(grip_norm, (0, CANONICAL_JOINTS["effector.position"] - grip_norm.shape[-1]))

        # Concatenate and pad to max_state_dim
        state = torch.cat([arm_padded, grip_padded], dim=-1)  # (14+2=16,)
        state = F.pad(state, (0, MAX_STATE_DIM - state.shape[-1]))  # (55,)

        # Build joint mask (which dims are real vs padding)
        state_joint_mask = torch.cat([
            F.pad(torch.ones(CANONICAL_JOINTS["arm.position"]), (0, CANONICAL_JOINTS["arm.position"] - 7)),
            F.pad(torch.ones(CANONICAL_JOINTS["effector.position"]), (0, CANONICAL_JOINTS["effector.position"] - 1)),
        ], dim=-1)
        state_joint_mask = F.pad(state_joint_mask, (0, MAX_STATE_DIM - state_joint_mask.shape[-1]))
        # Zero out padded dims
        state = state * state_joint_mask

        state = state.unsqueeze(0).to(device=self._device, dtype=torch.bfloat16)

        # --- Language ---
        language = obs["observation.language"]
        if isinstance(language, np.ndarray):
            language = str(language)
        prompt = language if language.startswith("<bos>") else f"<bos>{language}"
        if not prompt.endswith("\n"):
            prompt = f"{prompt}\n"

        tokenizer = self._processor.tokenizer
        tokenized = tokenizer(
            [prompt],
            padding="max_length",
            padding_side="right",
            max_length=72,
            truncation=True,
            return_tensors="pt",
        )
        lang_tokens = tokenized["input_ids"].to(device=self._device)
        lang_masks = tokenized["attention_mask"].to(device=self._device, dtype=torch.bool)

        return {
            "images": images,
            "img_masks": img_masks,
            "lang_tokens": lang_tokens,
            "lang_masks": lang_masks,
            "state": state,
            "image_grid_thw": image_grid_thw,
            # Also keep raw state for reverse transform
            "_raw_arm": arm_t,
            "_raw_grip": grip_t,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # Model output → competition action
    # ═══════════════════════════════════════════════════════════════════════

    def _model_to_action(
        self,
        canonical_actions: torch.Tensor,  # (1, 50, 55)
        model_inputs: dict,               # from _obs_to_model, includes _raw_arm, _raw_grip
    ) -> dict:
        """Convert canonical action tensor to competition-format action dict.

        Steps:
        1. Unpad canonical → real joint dims
        2. Unnormalize
        3. Add state back (reverse subtract_state for arm.position)
        4. Map to competition action keys
        """
        T = self.ACTION_CHUNK_SIZE  # 50

        # Squeeze batch dim
        actions = canonical_actions.squeeze(0).float()  # (50, 55)

        # Extract real dims using joint masks
        # arm.position: first 14 canonical dims, first 7 are real
        arm_canonical = actions[:, :CANONICAL_JOINTS["arm.position"]]  # (50, 14)
        arm_real = arm_canonical[:, :7]  # (50, 7)

        # effector.position: next 2 canonical dims, first 1 is real
        eff_start = CANONICAL_JOINTS["arm.position"]
        eff_canonical = actions[:, eff_start:eff_start + CANONICAL_JOINTS["effector.position"]]  # (50, 2)
        eff_real = eff_canonical[:, :1]  # (50, 1)

        # Unnormalize
        arm_unnorm = self._unnormalize(
            arm_real,
            self._norm_stats["action.arm.position"],
            norm_type="bounds_99_woclip",
        )
        eff_unnorm = self._unnormalize(
            eff_real,
            self._norm_stats["action.effector.position"],
            norm_type="bounds_99_woclip",
        )

        # Reverse subtract_state: action = model_output + state
        # (model was trained on action - state)
        raw_arm = model_inputs.get("_raw_arm", torch.zeros(7))
        if raw_arm.ndim == 0:
            raw_arm = torch.zeros(7)
        arm_absolute = arm_unnorm + raw_arm.to(arm_unnorm.device)  # (50, 7)

        # Build competition action dict
        action = {
            "meta.token":           self._token,
            "action.right_arm":     arm_absolute.cpu().numpy().astype(np.float32),
            "action.right_gripper": eff_unnorm.cpu().numpy().astype(np.float32),
            # Left arm/left gripper/pivot — hold at zero
            # (The model wasn't trained to control these)
            "action.left_arm":      np.zeros((T, 7), dtype=np.float32),
            "action.left_gripper":  np.zeros((T, 1), dtype=np.float32),
            "action.pivot":         np.zeros((T, 7), dtype=np.float32),
        }

        # Cache raw state for next step's reverse transform
        return action

    # ═══════════════════════════════════════════════════════════════════════
    # Normalization helpers
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _normalize(value: torch.Tensor, stats: dict, norm_type: str) -> torch.Tensor:
        if norm_type == "bounds_99_woclip":
            low = torch.as_tensor(stats["q01"], dtype=torch.float32)
            high = torch.as_tensor(stats["q99"], dtype=torch.float32)
            return (value - low) / (high - low + 1e-6) * 2.0 - 1.0
        elif norm_type == "meanstd":
            mean = torch.as_tensor(stats["mean"], dtype=torch.float32)
            std = torch.as_tensor(stats["std"], dtype=torch.float32)
            return (value - mean) / (std + 1e-6)
        else:
            return value

    @staticmethod
    def _unnormalize(value: torch.Tensor, stats: dict, norm_type: str) -> torch.Tensor:
        if norm_type == "bounds_99_woclip":
            low = torch.as_tensor(stats["q01"], dtype=torch.float32)
            high = torch.as_tensor(stats["q99"], dtype=torch.float32)
            return ((value + 1.0) / 2.0) * (high - low + 1e-6) + low
        elif norm_type == "meanstd":
            mean = torch.as_tensor(stats["mean"], dtype=torch.float32)
            std = torch.as_tensor(stats["std"], dtype=torch.float32)
            return value * (std + 1e-6) + mean
        else:
            return value
