"""
Convert a DCP-format LingBot-VLA checkpoint to a standard PeftModel format.

Usage:
    cd /mnt/ssd/lingbot-vla-v2
    source .venv/bin/activate
    python /home/zireael/unibot_submission/convert_checkpoint.py

This loads the DCP checkpoint via the training infrastructure and saves the
LoRA adapter weights as a HuggingFace-style peft checkpoint that can be
loaded with:
    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, lora_path)
"""

import os
import sys
from pathlib import Path

LINGBOT_ROOT = "/mnt/ssd/lingbot-vla-v2"
sys.path.insert(0, LINGBOT_ROOT)

import torch
import torch.nn as nn
from torch.distributed.checkpoint import FileSystemReader

from lingbotvla.models import build_foundation_model
from lingbotvla.utils.lora_utils import add_lora_to_model, freeze_parameters
from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config


def main():
    # Paths
    pretrained_path = "/mnt/ssd/lingbot-vla-v2/pretrained_models/lingbot-vla-v2-6b"
    tokenizer_path = "/mnt/ssd/lingbot-vla-v2/pretrained_models/Qwen3-VL-4B-Instruct"
    ckpt_path = Path("/home/zireael/lingbot_output/unibot_lora/checkpoints/global_step_4000/model")
    output_path = Path("/home/zireael/lingbot_output/unibot_lora/lora_adapter")

    print("Building FlowMatchingV2 config …")
    config = LingbotVLAV2Config(
        model_path=pretrained_path,
        tokenizer_path=tokenizer_path,
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
        router_activation="sigmoid",
        routed_scaling_factor=4.0,
        use_shared_expert_gate=False,
        action_dim=55,
        max_action_dim=55,
        max_state_dim=55,
        chunk_size=50,
        num_steps=10,
        freeze_vision_encoder=True,
        use_cache=True,
        enable_expert_vision=False,
        precompute_grid_thw=True,
        qwen3vl_use_vision_boundaries=True,
        tokenizer_max_length=72,
        vlm_causal=True,
        enable_fp32=False,
        expert_hidden_size=768,
        expert_intermediate_size=2752,
        action_num_attention_heads=32,
        action_num_key_value_heads=8,
        action_head_dim=128,
        align_params={},
        final_norm_adanorm=None,
    )

    print("Building foundation model …")
    model = build_foundation_model(
        config_path=None,
        config_cls=config,
        weights_path=pretrained_path,
        torch_dtype="bfloat16",
        init_device="cpu",
        force_use_huggingface=False,
        config_kwargs={},
        moe_implementation="eager",
    )

    print("Applying LoRA structure …")
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

    # Load DCP checkpoint via fallback loader
    print(f"Loading DCP checkpoint from {ckpt_path} …")
    distcp_files = sorted(ckpt_path.glob("__*_*.distcp"))
    if not distcp_files:
        raise FileNotFoundError(f"No .distcp files found in {ckpt_path}")

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
                    if dst.shape == tensor.shape:
                        dst.copy_(tensor)
                    else:
                        try:
                            dst.copy_(tensor.reshape(dst.shape))
                        except Exception:
                            pass
                    loaded += 1
        del shard

    print(f"Loaded {loaded} tensors from {len(distcp_files)} shards")

    # Save LoRA adapter
    print(f"Saving LoRA adapter to {output_path} …")
    output_path.mkdir(parents=True, exist_ok=True)

    # Find and save only LoRA parameters
    lora_state = {}
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            lora_state[name] = param.data.clone()

    torch.save(lora_state, output_path / "lora_weights.pt")
    print(f"Saved {len(lora_state)} LoRA parameters")

    # Also save the LoRA config for peft compatibility
    import json
    lora_config = {
        "lora_rank": 64,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj", "o_proj"],
        "task_type": "CAUSAL_LM",
    }
    with open(output_path / "lora_config.json", "w") as f:
        json.dump(lora_config, f, indent=2)

    print("Done! LoRA adapter saved.")


if __name__ == "__main__":
    main()
