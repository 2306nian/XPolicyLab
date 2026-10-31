# Pi_05

**Contributor:** RoboDojo Team | **Paper:** Pi0.5 technical report | **arXiv:** TBD | **Original code:** https://github.com/Physical-Intelligence/openpi

`Pi_05` adapts Physical Intelligence's π0.5 policy to XPolicyLab/RoboDojo through the uv-managed OpenPI stack. Integration scripts live at this directory level; the vendored upstream implementation lives in `openpi/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

```bash
cd XPolicyLab/policy/Pi_05
bash install.sh
source openpi/.venv/bin/activate  # OpenPI is uv-managed; there is no policy conda env
```

`eval.sh` arg 9 is not a conda env: pass `uv` (uses `deploy.yml` `policy_uv_env_path`) or an explicit OpenPI project path.

## Data Processing

Converts RoboDojo demonstrations into the LeRobot repo consumed by training. The dataset uses the official keys — `observation.state`, `action`, `observation.images.cam_high` / `cam_left_wrist` / `cam_right_wrist` ([official LeRobot conversion](../../README.md#official-lerobot-conversion)); the bundled script exists because conversion must run inside openpi's own pinned LeRobot environment, which sets the dataset version. The optional `expert_data_num` caps episodes for data conversion only (it is not part of checkpoint naming); the optional `raw_task_dirs` is a source task directory or comma-separated task list under `data/<bench_name>/` (defaults to `ckpt_name`). `raw_task_dirs` may also be passed directly as the 5th argument to write a differently named dataset from all of a task's demos, e.g. `bash process_data.sh RoboDojo stack_bowls_ablation arx_x5 joint stack_bowls`.

```bash
cd XPolicyLab/policy/Pi_05
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [raw_task_dirs]

# Example: convert stack_bowls demos for arx_x5 joint control
bash process_data.sh RoboDojo stack_bowls arx_x5 joint

# Example: create a 50-episode ablation while reading from the original task data
bash process_data.sh RoboDojo stack_bowls_50ep arx_x5 joint 50 stack_bowls
```

## Training

```bash
cd XPolicyLab/policy/Pi_05
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

# Example: train a cotrain run on GPU 0 (comma-separated gpu_id for multi-GPU)
bash train.sh RoboDojo cotrain arx_x5 joint 0 0
```

Checkpoints land in `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`; at eval time `ckpt_name` may be the short run name (auto-combined into that directory name), the full run-directory name, or a path to a checkpoint directory. By default training reads the LeRobot repo produced by `process_data.sh` (`<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>`); override with `OPENPI_LEROBOT_REPO_ID` when reusing an existing dataset. `train.sh` sets `fsdp_devices=1` for one visible GPU and `2` for multi-GPU by default (override with `OPENPI_FSDP_DEVICES`).

## Evaluation

```bash
cd XPolicyLab/policy/Pi_05
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_uv_env> <eval_env_conda_env>

# Example: evaluate a trained cotrain checkpoint on stack_bowls
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 0 0 uv <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## Configuration

`deploy.yml` keys to check before evaluation: `checkpoint_num`, `result_dir`, `obs_transform_pipeline`, `policy_uv_env_path`, `train_config_name` (must match the config used by `train.sh`), `repo_id`.

Set `inference_seed: 0` explicitly in deployment recipes (the adapter default is 0).
This integer in `[0, 2**32)` controls action-sampling noise independently of the
training/checkpoint naming field `seed`. Model initialization and every `reset()`
restart the noise sequence; consecutive inference calls advance it. JAX resets
its PRNG key without clearing compilation caches. PyTorch uses a separate
generator on the policy device and passes its noise through the sampler's existing
`noise` argument, leaving global RNGs unchanged. Explicitly supplied noise takes
precedence. This does not guarantee identical values across backends/devices or
deterministic device kernels. The effective seed is included in runtime metadata.
For ManiMux, preparing a new rollout sends RESET; Start/Resume and Pause/Home do
not reset this model RNG. If warmup inference is added, RESET after warmup before
the first measured request.

Environment variables used by the adapter scripts:

| Variable | Notes |
|---|---|
| `OPENPI_LEROBOT_REPO_ID` | Overrides the LeRobot repo id used by `train.sh`; defaults to `<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>`. |
| `OPENPI_FSDP_DEVICES` | Overrides the FSDP device count passed to OpenPI training. |
| `OPENPI_TRAIN_CONFIG_NAME` | Overrides the training config; defaults to `pi05_base_aloha_full_sim_arx-x5_seed_0`. |
| `OPENPI_DATA_MODE` | Data-processing mode passed to `openpi/scripts/process_data.py`; defaults to `image`. |
| `OPENPI_LOCAL_CACHE_ROOT` | Per-host local cache root for the HF datasets / JAX compilation caches; defaults to `/tmp/openpi-cache-$(hostname)`. |

`OPENPI_ROOT` and `OPENPI_SRC` are additional overrides consumed by the local scripts.

## Inference Sampling Capabilities

### Multi-environment default inference

`update_obs_batch` stores observations with distinct `env_idx` identities.
`get_action_batch(env_idx_list)` selects/reorders those rows and performs one
batched `Policy.infer` call, rather than one call per environment. Multi-env
requests support default sampling only; conditioned/multi-sample modes fail
explicitly. Single-env requests retain the existing sampling interface.

Optional noise is `[B, action_horizon, action_dim]` in requested-env order.
Empty requests return no actions; unknown/duplicate identities, malformed noise
and invalid/nonfinite model outputs are rejected. Batch shape changes can alter
BF16 kernel rounding and RNG consumption; this does not promise bitwise equality
with independently sampled single-env calls.

The optional deployment key `warmup_batch_sizes: [1, 2, 3, 4]` compiles these
shapes before serving, then restores the evaluation RNG and observation state.
`policy_timing.infer_ms` now includes device-to-host synchronization, reported by
`includes_device_to_host: true`.

CPU adapter regression (NumPy, Torch and einops; no checkpoint required):

```bash
python tests/unit/test_pi05_true_batch_adapter.py -v
```

The adapter implements the following WebSocket sampling modes. The handshake reports
only modes available for the loaded backend: PyTorch exposes `default`; JAX exposes
`default`, `rtc`, `aac` and `paint`, plus `autohorizon` with at least three denoising
steps. Per-request parameter validation still applies.

| Mode | Adapter method | OpenPI path |
|---|---|---|
| `default` | `get_action` | unchanged official single-sample inference |
| `rtc` | `get_action_rtc` | JAX Pi-guided conditioning hook |
| `aac` | `get_action_aac` | one prefix/KV-cache pass followed by an `N`-sample denoising batch |
| `paint` | `get_action_paint` | paper Algorithm 1: naive forward, backward Euler, prefix noise repaint, final forward |
| `autohorizon` | `get_action_autohorizon` | third-step action self-attention plus the pinned official bidirectional soft-pointer |

AAC accepts `{"mode": "aac", "num_samples": N}` for exactly one observation and returns `N`
native action chunks. It is JAX-only and cannot be combined with RTC conditioning. The adapter does
not calculate entropy, robot kinematics, motion thresholds or candidate selection; those remain
client/runtime responsibilities. Calls that omit AAC parameters continue through the original
single-sample callable.

PAINT accepts `{"mode": "paint", "action_prefix": A[s:s+d], "delay_steps": d}`. The adapter
normalizes the raw robot-unit prefix with the same official input transform as model actions, then
runs `3N` velocity evaluations without gradients. The public PAINT repository currently contains
documentation rather than source code, so this path is an explicit paper reproduction of
arXiv:2606.19774, not an upstream-code claim.

## Tianji pass-ball zero-pose deployment

This opt-in, deployment-only profile supports `env_cfg_type: tianji_dual` and `action_type: ee`.
Select `observation_profile: tianji_taccap_pi05_zero_pose`; existing ALOHA/YAM profiles are
unchanged. `model.py` constructs the dedicated `PassBallZeroPoseModel(ModelTemplate)` for
this profile, using the same shared XPolicyLab server and checkpoint/dimension helpers.

Source: Physical Intelligence OpenPI, https://github.com/Physical-Intelligence/openpi,
training revision `215abfb217dbac7d5f1273282331b9b1866c0479`. The local pass-ball reproduction
comes from the submitted source archive of run
`pi05-passball-zero-pose-h32-b16-60k-20260923-192017`; it has no separate upstream repository
URL. `pass_ball_pose.py` and `pass_ball_state.py` retain the submitted numeric transforms
and their source SHA256 comments. `pass_ball_model.py` reconstructs its inference transforms
without importing the private training project or referring to training-machine paths.
OpenPI remains in the existing vendored `openpi/` directory with its license intact.

This profile's data conversion and training entry points are **not integrated** into the
standard `process_data.sh` / `train.sh`; those existing commands still serve their original
profiles. The downloaded training archive is provenance, not a runtime dependency.

Required deployment keys: `model_state_encoding: zero_pose`,
`train_config_name: pi05_pass_ball_hifi_umi_lora_zero_pose`, `action_horizon: 32`,
`output_format: xpolicylab`, `action_semantics: absolute_per_arm_base_xyz_wxyz`,
`checkpoint_num`, `checkpoint_variant`, `checkpoint_source`, `model_path`,
`norm_stats_path`, `norm_stats_sha256`, `norm_stats_source`, `repo_id`, `num_steps`.
The model is LoRA Pi05 with 20 native dimensions padded to 32. The native order is
right/left with Rot6D first-two-row rotations and continuous normalized openings.
Model pose slots are zeroed after quantile normalization; real request TCPs remain
outside the model as the shared anchor for all 32 targets. The tool-axis change is
inverted before returning standard absolute per-arm-base `xyz + wxyz` action dictionaries.
Two wrist RGB images are active; the synthetic black base view has a false image mask.
Only default sampling is supported for this profile.

Run one hardware-free, synthetic-image forward from the parent workspace:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  -m XPolicyLab.policy.Pi_05.offline_pass_ball \
  --config <absolute-zero-pose-recipe.yaml> --checkpoint <export>/checkpoints/20000
```

The export directory name, step and
normalization SHA256 must match the selected recipe. See
`.agents/skills/manimux-inference/references/deployment/pi05-tianji-taccap.md` in the parent ManiMux workspace for paired recipes,
station binding, artifact checking and full service commands.

## Tianji pack-plate wrist-only checkpoint

This is a separate deployment profile, `tianji_taccap_pi05_pack_plate`, selected only
by the pack-plate recipe. It uses the verified `checkpoint-59999` export under
`pi05-pack-plate-wrist-only-final-59999` and its own
`assets/pack-plate-taccap-h32-zero-pose/norm_stats.json`. The LeRobot task string
is `plate`. Two wrist RGB images are active; the synthetic base image is masked.
The 20-D native order, zero-pose state, current-TCP-relative 32-step actions and
absolute normalized grippers use the Tianji transform helpers above. The local
`train_config_name` is a deployment reconstruction; the export does not contain
the submitted training configuration. Data conversion and training entry points
for this specific checkpoint are not included.

The `pi05-pack-plate-full-zero-state-pack-instruction-final-59999` export uses
the same deployment transforms with full-model weights. Its recipe explicitly
selects `paligemma_variant: gemma_2b` and `action_expert_variant: gemma_300m`;
it has no LoRA parameter leaves. Its normalization asset is
`pack-plate-taccap-h32-zero-pose-pack-instruction`. The old wrist-only export
continues to use `gemma_2b_lora` and `gemma_300m_lora` by default.

This pack-plate profile supports default and RTC sampling. The RTC condition
arrives as left/right absolute TCP poses and normalized openings from ManiMux;
it is converted to the model's right/left 20-D absolute layout, then the
existing input transform makes every conditioned pose relative to the current
observation before normalization and Pi-guided JAX sampling. Zero-weight rows
use the observation anchor as a valid pose. The pass-ball profile remains
default-only; PAINT is not advertised for pack-plate.

Static artifact inspection uses `manimux.servers.pi05 --check` with the paired
experiment and an isolated station. A synthetic GPU forward, when the GPU is idle:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  -m XPolicyLab.policy.Pi_05.offline_pack_plate \
  --config manimux/configs/policy/pi05/tianji/pack_plate/wrist-only-step59999.yaml \
  --checkpoint <export>/checkpoint-59999
```

The exported README does not specify the training-time TCP axis conversion.
The deployment currently shares the tool-axis transform used for the pass-ball
Tianji data. Confirm it against the pack-plate training source before physical
execution. See `.agents/skills/manimux-inference/references/deployment/pi05-tianji-pack-plate.md` in the parent workspace.

## Tianji pack-plate inter-hand-relative checkpoints

The tracker-fixed export adds independently selectable step 20000 (best validation)
and step 59999 (final) **LoRA** recipes. They select `model_state_encoding:
inter_hand_relative` and provide `state_base_to_common` as left/right rigid 4x4
transforms in metres. The transforms are part of the reported backend identity.

`pack_plate_state.InterHandRelativeState` converts the per-arm TCPs into a common
frame and encodes each hand in the other hand's training tool frame. It runs after
`PassBallInputs` has formed relative RTC targets and before state normalization.
Unlike zero-pose recipes, this variant preserves normalized pose values during
tokenization. Original per-arm observation anchors, action decoding and the RTC
sampler remain shared. Existing zero-pose recipes keep their previous transforms.

Each recipe pins its own checkpoint-local normalization artifact. The deployment
uses the existing `PackPlateZeroPoseModel` class and server dispatch; its historical
class name does not choose the state representation. See
[the independent experiment runbook](../../../.agents/skills/manimux-inference/references/deployment/pi05-tianji-pack-plate-inter-hand-relative.md)
for geometry provenance, frozen settings, launch commands and validation status.
