# UniBot V1 Challenge — LingBot-VLA 2.0 LoRA Submission

**Team:** ezhoureal  
**Model:** LingBot-VLA 2.0 (6.3B parameters) — Qwen3-VL-4B-Instruct backbone + Qwen2 action expert with Mixture-of-Experts layers  
**Fine-tuning:** LoRA rank=64 on q_proj/v_proj/o_proj attention projections (65M trainable / 6.3B total, 1.04%), merged into base weights for deployment  
**Hardware:** Single RTX 4090 (24GB VRAM)  
**Control space:** `joint` — left arm (7) + right arm (7) + left/right gripper (1+1) + pivot (7)

## Quick Start

Everything below works from a fresh machine: this git repo, the
lingbot-vla-v2 git repo, and the HuggingFace weights are the only inputs.
No absolute paths are required — the defaults expect `lingbot-vla-v2/` and
`unibot_weights/` inside this submission directory (override with
`LINGBOT_ROOT` / `WEIGHTS_ROOT` / `MODEL_PATH` / `NORM_STATS_PATH`).

### 1. Code

```bash
git clone https://github.com/ezhoureal/unibot_submission
cd unibot_submission

# lingbot-vla-v2 codebase (model code, deploy server, robot configs)
git clone https://github.com/ezhoureal/lingbot-vla-v2
pip install -e ./lingbot-vla-v2
pip install -r requirements.txt
```

### 2. Model weights (HuggingFace)

```bash
pip install -U "huggingface_hub[cli]"
hf download ezhoureal/lingbot-vla-v2-unibot-lora --local-dir unibot_weights
```

This downloads the merged full-model checkpoint (~12.5 GB):

```
unibot_weights/
├── checkpoints/global_step_step=10000/model/model.safetensors
├── lingbotvla_cli.yaml        # training config (paths sanitized)
└── norm_stats/unibot_full.json
```

The tokenizer/processor is fetched automatically from
`Qwen/Qwen3-VL-4B-Instruct` on first load; the base model weights are already
merged into `model.safetensors` — no other downloads are needed.

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

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `LINGBOT_ROOT` | `./lingbot-vla-v2` | lingbot-vla-v2 repo checkout |
| `WEIGHTS_ROOT` | `./unibot_weights` | HuggingFace weights download |
| `MODEL_PATH` | `$WEIGHTS_ROOT/checkpoints/global_step_step=10000/model` | merged safetensors dir |
| `NORM_STATS_PATH` | `$WEIGHTS_ROOT/norm_stats/unibot_full.json` | normalization stats |
| `ROBOT_NAME` | `unibot/g1_dex1_full` | robot config under `configs/robot_configs/` |
| `UNIBOT_SUBMISSION_TOKEN` | — (required) | competition token |
| `UNIBOT_CONTROL_SPACE` | `joint` | only `joint` is supported |

## Re-training / re-export (optional)

To reproduce the weights from your own training run (in the lingbot-vla-v2
repo):

```bash
bash train.sh scripts/train_unibot_lora.py configs/vla/unibot/unibot_lora.yaml \
    --train.output_dir output/unibot_lora_full

python scripts/export_unibot_lora_merged.py \
    --ckpt output/unibot_lora_full/checkpoints/global_step_step=10000.ckpt \
    --output-dir output/unibot_lora_full/checkpoints/global_step_step=10000/model
```

## Files

| File | Role |
|------|------|
| `lingbot_policy.py` | Competition policy — thin adapter over `deploy.lingbot_vla_v2_policy.LingbotVLAv2Server` (training-exact feature transforms) |
| `run_server.py` | Server entry point — wraps the policy in `PolicyService` |
| `run_client.py` | Local evaluator stub (from template) |
| `example_env.py` | Validates action format (from template) |
| `policy/` | WebSocket transport layer (`PolicyService` / `RemotePolicy`) |
| `requirements.txt` | Python dependencies |

Checkpoint export lives in the lingbot-vla-v2 repo: `scripts/export_unibot_lora_merged.py`.

## Observation / Action Mapping

| Competition key | Dataset feature (canonical slot) |
|-----------------|----------------------------------|
| `observation.images.cam_left_high` | `head_stereo_left` → `camera_top` |
| `observation.images.cam_right_high` | `head_stereo_right` → `camera_top_right` |
| `observation.images.cam_left_wrist` | `wrist_left` → `camera_wrist_left` |
| `observation.images.cam_right_wrist` | `wrist_right` → `camera_wrist_right` |
| `observation.state.left_arm` / `right_arm` | `arm.position[0:7]` / `[7:14]` |
| `observation.state.left/right_gripper` | `effector.position[0]` / `[1]` |
| `observation.state.lower_body[12:15]` | `waist.position[0:3]` (state) |
| `action.left_arm` / `right_arm` | `arm.position` (delta → + current state) |
| `action.left/right_gripper` | `effector.position` |
| `action.pivot[0:3]` | `base.position` (vx, vy, angle_z) |
| `action.pivot[6]` | `waist.position[3]` (body height) |

## Training Summary

- **Dataset:** UniBot G1_Dex1 full — all 32 tasks, 33,276 episodes
- **Training:** 10K steps, bf16 model + fp32 LoRA, single RTX 4090
- **Loss:** val/loss 0.085 at the final checkpoint
- **Normalization:** meanstd on actions (and states) per `assets/norm_stats/unibot_full.json`; `arm.position` actions are state-relative deltas (`subtract_state`)

## Limitations

- **Joint control space only** — the checkpoint was not trained on EE-pose actions
- `pivot[3:6]` carry no learnable signal and were dropped in training; the policy re-fills them with the dataset constants `(0, 0.149, 0)`
- Waist yaw/roll/pitch actions are predicted but not part of the competition action spec, so they are not sent (the pivot height and base commands are)
- Flow-matching inference runs ~10 denoising steps per `get_action` call
