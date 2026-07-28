# UniBot V1 Challenge — LingBot-VLA 2.0 LoRA Submission

**Team:** ezhoureal  
**Model:** LingBot-VLA 2.0 (6.3B parameters) — Qwen3-VL-4B-Instruct backbone + Qwen2 action expert with Mixture-of-Experts layers  
**Fine-tuning:** LoRA rank=64 on q_proj/v_proj/o_proj attention projections (65M trainable / 6.3B total, 1.04%)  
**Hardware:** Single RTX 4090 (24GB VRAM)  
**Control space:** `joint` — right arm (7 DOF) + right gripper (1 DOF)

## Quick Start

### 1. Environment Setup

```bash
# Clone the lingbot-vla-v2 codebase (required dependency)
git clone https://github.com/ezhoureal/lingbot-vla-v2  # or your fork
cd lingbot-vla-v2
pip install -e .

# Install submission dependencies
cd /path/to/unibot_submission
pip install -r requirements.txt
```

### 2. Model Weights

Set these environment variables to point at your model files:

```bash
export PRETRAINED_MODEL_PATH=/path/to/lingbot-vla-v2-6b
export TOKENIZER_PATH=/path/to/Qwen3-VL-4B-Instruct
export LORA_CHECKPOINT_PATH=/path/to/checkpoints/global_step_4000/model
export NORM_STATS_PATH=/path/to/norm_stats/g1_dex1.json
export LINGBOT_ROOT=/path/to/lingbot-vla-v2
```

### 3. Local Verification

```bash
# Terminal A — serve the policy
UNIBOT_SUBMISSION_TOKEN=dev-token UNIBOT_CONTROL_SPACE=joint python run_server.py 8765

# Terminal B — run the evaluator stub
UNIBOT_SUBMISSION_TOKEN=dev-token python run_client.py
```

### 4. Deployment

```bash
# Set the organiser-issued token
export UNIBOT_SUBMISSION_TOKEN=<your-competition-token>

# Serve on N consecutive ports (one per evaluation robot, e.g. N=3)
UNIBOT_SUBMISSION_TOKEN=$UNIBOT_SUBMISSION_TOKEN UNIBOT_CONTROL_SPACE=joint python run_server.py 8765 &
UNIBOT_SUBMISSION_TOKEN=$UNIBOT_SUBMISSION_TOKEN UNIBOT_CONTROL_SPACE=joint python run_server.py 8766 &
UNIBOT_SUBMISSION_TOKEN=$UNIBOT_SUBMISSION_TOKEN UNIBOT_CONTROL_SPACE=joint python run_server.py 8767 &
```

## Files

| File | Role |
|------|------|
| `lingbot_policy.py` | LingBot-VLA 2.0 + LoRA policy implementing the §1 interface |
| `run_server.py` | Server entry point — wraps the policy in `PolicyService` |
| `run_client.py` | Local evaluator stub (from template) |
| `example_env.py` | Validates action format (from template) |
| `convert_checkpoint.py` | Utility: convert DCP checkpoint → standalone LoRA weights |
| `policy/` | WebSocket transport layer (`PolicyService` / `RemotePolicy`) |
| `requirements.txt` | Python dependencies |

## Training Summary

- **Dataset:** UniBot G1_Dex1_ArrangePlates (175 valid episodes, 165K frames)
- **Training:** ~6K steps, ~5 hours on RTX 4090
- **Loss:** 0.50 → 0.12 (76% reduction)
- **VRAM:** 12.7 GB peak (bf16 model)

## Limitations (Baseline)

This is a **competition baseline** submission:

- Only controls **right arm + right gripper** (left arm, left gripper, and pivot return zero actions)
- Uses a **single camera** (cam_left_high mapped to head_stereo_left)
- Trained on **1 of 32 tasks** (ArrangePlates) due to incomplete dataset downloads
- Flow-matching inference takes ~10 denoising steps per `get_action` call
