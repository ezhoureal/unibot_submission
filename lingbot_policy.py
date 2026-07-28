"""
LingBot-VLA 2.0 LoRA policy for UniBot V1 Challenge.

Implements the competition policy interface:
  - metadata      → dict describing obs/action contract
  - get_action(obs) → action chunk dict
  - reset()       → clear per-episode state

Architecture: LingBot-VLA 2.0 (6.3B): Qwen3-VL-4B-Instruct backbone + Qwen2
action expert (36 MoE layers). LoRA rank=64 on q_proj/v_proj/o_proj.
Fine-tuned on UniBot ArrangePlates task (right arm + gripper, single camera).

Control space: "joint" — right arm (7 DOF) + right gripper (1 DOF).
Left arm, left gripper, and pivot return zeros (model not trained on these).

Requirements:
  - Single RTX 4090 (24GB VRAM)
  - lingbot-vla-v2 repo on PYTHONPATH
  - DCP checkpoint at LORA_CHECKPOINT_PATH
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
LINGBOT_ROOT = os.environ.get("LINGBOT_ROOT", "/mnt/ssd/lingbot-vla-v2")
sys.path.insert(0, LINGBOT_ROOT)

PRETRAINED_MODEL_PATH = os.environ.get(
    "PRETRAINED_MODEL_PATH", "/mnt/ssd/lingbot-vla-v2/pretrained_models/lingbot-vla-v2-6b"
)
TOKENIZER_PATH = os.environ.get(
    "TOKENIZER_PATH", "/mnt/ssd/lingbot-vla-v2/pretrained_models/Qwen3-VL-4B-Instruct"
)
LORA_CHECKPOINT_PATH = os.environ.get(
    "LORA_CHECKPOINT_PATH",
    "/home/zireael/lingbot_output/unibot_lora/checkpoints/global_step_4000/model",
)
NORM_STATS_PATH = os.environ.get(
    "NORM_STATS_PATH", "/home/zireael/lingbot_output/norm_stats/g1_dex1.json"
)

# ── Canonical space dimensions ─────────────────────────────────────────────
CANONICAL_JOINTS = {"arm.position": 14, "effector.position": 2}
CANONICAL_ORDER = ["arm.position", "effector.position"]
MAX_ACTION_DIM = 55
MAX_STATE_DIM = 55
N_ACTION_STEPS = 50
NUM_DENOISE_STEPS = 10


class LingBotPolicy:
    """UniBot-V1 policy backed by a LoRA-fine-tuned LingBot-VLA 2.0 model."""

    OBS_CHUNK_SIZE = 1
    ACTION_CHUNK_SIZE = 50
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

        # Initialize torch.distributed for DCP checkpoint loading
        self._init_distributed()

        log.info("Loading LingBot-VLA 2.0 with LoRA …")
        self._load_model()
        self._load_norm_stats()
        log.info(
            "LingBotPolicy ready (%.1fM trainable / %.1fM total params)",
            sum(p.numel() for p in self._model.parameters() if p.requires_grad) / 1e6,
            sum(p.numel() for p in self._model.parameters()) / 1e6,
        )

    def _init_distributed(self):
        """Initialize single-process distributed for DCP checkpoint loading."""
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo", rank=0, world_size=1)

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
        t_start = time.perf_counter()

        # 1. Convert competition observation → model inputs
        model_inputs = self._obs_to_model(obs)

        # 2. Run flow-matching inference
        with torch.inference_mode():
            canonical_actions = self._sample_actions_reliable(
                images=model_inputs["images"],
                img_masks=model_inputs["img_masks"],
                lang_tokens=model_inputs["lang_tokens"],
                lang_masks=model_inputs["lang_masks"],
                state=model_inputs["state"],
                image_grid_thw=model_inputs["image_grid_thw"],
            )  # (1, 50, 55)

        # 3. Convert canonical action → competition action dict
        action = self._model_to_action(canonical_actions, model_inputs)

        elapsed = time.perf_counter() - t_start
        self._step += 1
        log.info("[step %d] inference %.2fs", self._step, elapsed)
        return action

    def reset(self):
        self._step = 0
        return {"ok": True}

    # ═══════════════════════════════════════════════════════════════════════
    # Model loading
    # ═══════════════════════════════════════════════════════════════════════

    def _load_model(self):
        """Build model, apply LoRA, load DCP checkpoint, move to GPU."""
        from lingbotvla.models import build_foundation_model, build_processor
        from lingbotvla.utils.lora_utils import add_lora_to_model
        from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import (
            LingbotVLAV2Config,
        )

        config = LingbotVLAV2Config(
            model_path=PRETRAINED_MODEL_PATH,
            tokenizer_path=TOKENIZER_PATH,
            moe_implementation="eager",
            vit_attn_implementation="sdpa",
            attention_implementation="eager",
            post_training=False,
            adanorm_time=True,
            use_moe=True,
            token_moe_layers=list(range(36)),
            token_num_experts=32,
            token_top_k=4,
            token_moe_intermediate_size=512,
            token_shared_intermediate_size=704,
            bias_update_speed=0,
            sequence_wise_loss_coeff=0,  # ← must be 0 to avoid Triton
            router_z_loss_coeff=0,
            router_activation="sigmoid",
            routed_scaling_factor=4.0,
            use_shared_expert_gate=False,
            action_dim=MAX_ACTION_DIM,
            max_action_dim=MAX_ACTION_DIM,
            max_state_dim=MAX_STATE_DIM,
            chunk_size=N_ACTION_STEPS,
            num_steps=NUM_DENOISE_STEPS,
            freeze_vision_encoder=True,
            use_cache=False,  # ← use training-style forward (no KV cache)
            enable_expert_vision=False,
            precompute_grid_thw=True,
            qwen3vl_use_vision_boundaries=True,
            tokenizer_max_length=72,
            use_qwen3_chat_template=False,
            vlm_causal=True,
            enable_fp32=False,
            expert_hidden_size=768,
            expert_intermediate_size=2752,
            action_num_attention_heads=32,
            action_num_key_value_heads=8,
            action_head_dim=128,
            align_params={},
        )

        log.info("Building foundation model …")
        model = build_foundation_model(
            config_path="qwen3vl",
            config_cls=config,
            weights_path=PRETRAINED_MODEL_PATH,
            torch_dtype="bfloat16",
            init_device="cpu",
            force_use_huggingface=False,
            moe_implementation="eager",
            config_kwargs={
                "tokenizer_path": TOKENIZER_PATH,
                "post_training": False,
                "adanorm_time": True,
            },
        )

        log.info("Applying LoRA (rank=64) …")
        lora_target_modules_support = set()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                lora_target_modules_support.add(name.split(".")[-1])
        add_lora_to_model(
            model,
            lora_rank=64,
            lora_alpha=32,
            lora_target_modules="q_proj,v_proj,o_proj",
            init_lora_weights="kaiming",
            lora_target_modules_support=lora_target_modules_support,
        )

        # Load DCP checkpoint
        ckpt_path = Path(LORA_CHECKPOINT_PATH)
        if ckpt_path.exists():
            log.info("Loading LoRA checkpoint …")
            self._load_dcp_checkpoint(model, ckpt_path)
        else:
            log.warning("Checkpoint %s not found — base model only", ckpt_path)

        # Move to GPU
        model = model.to(dtype=torch.bfloat16).cuda()
        model.eval()
        self._model = model

        # Load processor
        log.info("Loading Qwen3-VL processor …")
        self._processor = build_processor(TOKENIZER_PATH)
        self._image_processor = self._processor.image_processor
        self._tokenizer = self._processor.tokenizer

    def _load_dcp_checkpoint(self, model, ckpt_path: Path):
        """Load DCP checkpoint with key prefix mapping."""
        from torch.distributed.checkpoint import FileSystemReader, load as dcp_load
        from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

        reader = FileSystemReader(ckpt_path)
        metadata = reader.read_metadata()
        model_sd = model.state_dict()

        alias = {}
        for ckpt_key in metadata.state_dict_metadata:
            if ckpt_key.startswith("state.model.model."):
                model_key = "model." + ckpt_key[len("state.model.model."):]
                if model_key in model_sd:
                    alias[ckpt_key] = model_sd[model_key]

        dcp_load(
            state_dict=alias,
            checkpoint_id=ckpt_path,
            storage_reader=reader,
            planner=DefaultLoadPlanner(allow_partial_load=True),
        )
        log.info("DCP checkpoint loaded (%d keys)", len(alias))

    def _load_norm_stats(self):
        with open(NORM_STATS_PATH) as f:
            self._norm_stats = json.load(f)["norm_stats"]

    # ═══════════════════════════════════════════════════════════════════════
    # Reliable flow-matching inference (training-style forward, no KV cache)
    # ═══════════════════════════════════════════════════════════════════════

    def _sample_actions_reliable(
        self, images, img_masks, lang_tokens, lang_masks, state, image_grid_thw
    ) -> torch.Tensor:
        """Run flow-matching denoising using the training-style forward path.

        Unlike the original ``sample_actions`` which uses KV-cache based
        two-stage inference, this method processes prefix+suffix together
        in each denoising step. It is ~2× slower but avoids an attention-mask
        compatibility issue with the eager attention implementation.

        Returns:
            actions: (1, n_action_steps, max_action_dim) in canonical space
        """
        from lingbotvla.models.vla.lingbot_vla.utils import make_att_2d_masks

        fm = self._model.model  # FlowMatchingV2
        bsize = state.shape[0]
        dtype = state.dtype
        device = state.device

        # Embed prefix once
        (
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            prefix_position_ids,
            visual_pos_masks,
            deepstack_embs,
        ) = fm.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, image_grid_thw=image_grid_thw
        )

        # Flow matching ODE integration
        actions_shape = (bsize, N_ACTION_STEPS, MAX_ACTION_DIM)
        x_t = torch.randn(actions_shape, device=device, dtype=dtype)
        dt = -1.0 / NUM_DENOISE_STEPS
        time_val = 1.0

        for _ in range(NUM_DENOISE_STEPS):
            time_tensor = torch.full((bsize,), time_val, device=device, dtype=torch.float32)

            # Embed suffix
            time_embs, suffix_embs, suffix_pad_masks, suffix_att_masks = fm.embed_suffix(
                state, x_t, time_tensor
            )

            # Build combined masks
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            position_ids = fm._build_full_position_ids(
                prefix_position_ids, prefix_pad_masks, suffix_pad_masks
            )

            # Full forward (prefix + suffix together)
            out_embs, _, _ = fm.qwenvl_with_expert.forward(
                attention_mask=att_2d_masks,
                position_ids=position_ids,
                vlm_position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                fill_kv_cache=False,
                ada_cond=time_embs
                if getattr(fm.config, "adanorm_time", False)
                else None,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_embs,
            )

            # Extract suffix output and project to velocity
            suffix_out = out_embs[1][:, -N_ACTION_STEPS:]
            if getattr(fm.config, "action_fp32", False):
                v_t = fm._fp32_linear(fm.action_out_proj, suffix_out)
            else:
                v_t = fm.action_out_proj(suffix_out.to(fm.action_out_proj.weight.dtype))

            # ODE step: x_t += dt * v_t
            x_t = x_t + dt * v_t.to(dtype=dtype)
            time_val += dt

        return x_t

    # ═══════════════════════════════════════════════════════════════════════
    # Observation → model input
    # ═══════════════════════════════════════════════════════════════════════

    def _obs_to_model(self, obs: dict) -> dict:
        """Convert competition-format observation to model inputs."""
        # --- Image ---
        raw_img = obs["observation.images.cam_left_high"]
        if raw_img.ndim > 3:
            raw_img = raw_img[-1]  # (T, H, W, 3) → (H, W, 3)

        if isinstance(raw_img, np.ndarray):
            img_tensor = torch.from_numpy(raw_img).float()
        else:
            img_tensor = raw_img.float()

        # Resize to 224×224
        img_tensor = img_tensor.permute(2, 0, 1)  # HWC → CHW
        img_tensor = F.interpolate(
            img_tensor.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
        ).squeeze(0)
        img_tensor = img_tensor.clamp(0, 255).to(torch.uint8)

        processed = self._image_processor(img_tensor)
        images = processed["pixel_values"].unsqueeze(0).cuda()
        image_grid_thw = processed.get("image_grid_thw")
        if image_grid_thw is not None:
            if image_grid_thw.ndim == 1:
                image_grid_thw = image_grid_thw.unsqueeze(0)
            image_grid_thw = image_grid_thw.cuda()

        img_masks = torch.ones(1, 1, dtype=torch.bool, device=self._device)

        # --- State ---
        right_arm = obs["observation.state.right_arm"]
        if right_arm.ndim > 1:
            right_arm = right_arm[-1]
        right_gripper = obs["observation.state.right_gripper"]
        if right_gripper.ndim > 1:
            right_gripper = right_gripper[-1]

        arm_t = torch.as_tensor(right_arm, dtype=torch.float32)
        grip_t = torch.as_tensor(right_gripper, dtype=torch.float32)

        # Normalize
        arm_norm = self._normalize(
            arm_t,
            self._norm_stats["observation.state.arm.position"],
            "bounds_99_woclip",
        )
        grip_norm = self._normalize(
            grip_t,
            self._norm_stats["observation.state.effector.position"],
            "bounds_99_woclip",
        )

        # Pad to canonical dims
        arm_padded = F.pad(arm_norm, (0, CANONICAL_JOINTS["arm.position"] - 7))
        grip_padded = F.pad(grip_norm, (0, CANONICAL_JOINTS["effector.position"] - 1))

        # Build joint mask and padded state
        state_joint = torch.cat([arm_padded, grip_padded], dim=-1)
        state_padded = F.pad(state_joint, (0, MAX_STATE_DIM - state_joint.shape[-1]))

        # Zero out padded dims
        joint_mask = torch.cat([
            F.pad(torch.ones(7), (0, CANONICAL_JOINTS["arm.position"] - 7)),
            F.pad(torch.ones(1), (0, CANONICAL_JOINTS["effector.position"] - 1)),
        ], dim=-1)
        joint_mask = F.pad(joint_mask, (0, MAX_STATE_DIM - joint_mask.shape[-1]))
        state_padded = state_padded * joint_mask

        state = state_padded.unsqueeze(0).to(device=self._device, dtype=torch.bfloat16)

        # --- Language ---
        language = obs["observation.language"]
        if isinstance(language, np.ndarray):
            language = str(language)
        prompt = language if language.startswith("<bos>") else f"<bos>{language}"
        if not prompt.endswith("\n"):
            prompt = f"{prompt}\n"

        tokenized = self._tokenizer(
            [prompt],
            padding="max_length",
            padding_side="right",
            max_length=72,
            truncation=True,
            return_tensors="pt",
        )
        lang_tokens = tokenized["input_ids"].to(device=self._device)
        lang_masks = tokenized["attention_mask"].to(
            device=self._device, dtype=torch.bool
        )

        return {
            "images": images,
            "img_masks": img_masks,
            "lang_tokens": lang_tokens,
            "lang_masks": lang_masks,
            "state": state,
            "image_grid_thw": image_grid_thw,
            "_raw_arm": arm_t,
            "_raw_grip": grip_t,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # Model output → competition action
    # ═══════════════════════════════════════════════════════════════════════

    def _model_to_action(
        self, canonical_actions: torch.Tensor, model_inputs: dict
    ) -> dict:
        """Convert canonical action tensor to competition-format action dict."""
        T = self.ACTION_CHUNK_SIZE

        actions = canonical_actions.squeeze(0).float()  # (50, 55)

        # Extract real dims
        arm_canonical = actions[:, : CANONICAL_JOINTS["arm.position"]]
        arm_real = arm_canonical[:, :7]

        eff_start = CANONICAL_JOINTS["arm.position"]
        eff_canonical = actions[
            :, eff_start : eff_start + CANONICAL_JOINTS["effector.position"]
        ]
        eff_real = eff_canonical[:, :1]

        # Unnormalize
        arm_unnorm = self._unnormalize(
            arm_real,
            self._norm_stats["action.arm.position"],
            "bounds_99_woclip",
        )
        eff_unnorm = self._unnormalize(
            eff_real,
            self._norm_stats["action.effector.position"],
            "bounds_99_woclip",
        )

        # Reverse subtract_state: action_absolute = delta + state
        raw_arm = model_inputs.get("_raw_arm", torch.zeros(7))
        arm_absolute = arm_unnorm + raw_arm.to(arm_unnorm.device)

        # Build competition action dict
        action = {
            "meta.token": self._token,
            "action.right_arm": arm_absolute.cpu().numpy().astype(np.float32),
            "action.right_gripper": eff_unnorm.cpu().numpy().astype(np.float32),
            "action.left_arm": np.zeros((T, 7), dtype=np.float32),
            "action.left_gripper": np.zeros((T, 1), dtype=np.float32),
            "action.pivot": np.zeros((T, 7), dtype=np.float32),
        }
        return action

    # ═══════════════════════════════════════════════════════════════════════
    # Normalization helpers
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _normalize(value: torch.Tensor, stats: dict, norm_type: str) -> torch.Tensor:
        device = value.device
        if norm_type == "bounds_99_woclip":
            low = torch.as_tensor(stats["q01"], dtype=torch.float32, device=device)
            high = torch.as_tensor(stats["q99"], dtype=torch.float32, device=device)
            return (value - low) / (high - low + 1e-6) * 2.0 - 1.0
        elif norm_type == "meanstd":
            mean = torch.as_tensor(stats["mean"], dtype=torch.float32, device=device)
            std = torch.as_tensor(stats["std"], dtype=torch.float32, device=device)
            return (value - mean) / (std + 1e-6)
        return value

    @staticmethod
    def _unnormalize(value: torch.Tensor, stats: dict, norm_type: str) -> torch.Tensor:
        device = value.device
        if norm_type == "bounds_99_woclip":
            low = torch.as_tensor(stats["q01"], dtype=torch.float32, device=device)
            high = torch.as_tensor(stats["q99"], dtype=torch.float32, device=device)
            return ((value + 1.0) / 2.0) * (high - low + 1e-6) + low
        elif norm_type == "meanstd":
            mean = torch.as_tensor(stats["mean"], dtype=torch.float32, device=device)
            std = torch.as_tensor(stats["std"], dtype=torch.float32, device=device)
            return value * (std + 1e-6) + mean
        return value
