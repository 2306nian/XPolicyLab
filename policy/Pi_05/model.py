#!/usr/bin/env python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import dataclasses
import math
import sys
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any

import numpy as np
from openpi.policies import policy_config as _policy_config
from openpi.shared import normalize as _normalize
from openpi.training import config as _config

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import candidate_checkpoint_roots
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)

_POLICY_DIR = Path(__file__).resolve().parent
_CHECKPOINTS_DIR = _POLICY_DIR / "checkpoints"


def _extract_step_number(value: Any) -> int | None:
    matches = [part for part in str(value).split("/") if part]
    if not matches:
        return None
    digits = "".join(ch for ch in matches[-1] if ch.isdigit())
    return int(digits) if digits else None


def _resolve_pi05_model_root(model_cfg: dict[str, Any]) -> Path:
    # Shared precedence: model_path/checkpoint_path keys > ckpt_name-as-path >
    # {bench}-{ckpt}-{env}-{action}-{seed} concat > checkpoints/<ckpt_name>.
    candidates = candidate_checkpoint_roots(
        model_cfg,
        _CHECKPOINTS_DIR,
        policy_dir=_POLICY_DIR,
        explicit_keys=("model_path", "checkpoint_path"),
    )
    if not candidates:
        raise ValueError("ckpt_name or model_path is required for Pi_05.")
    checkpoint_root = next(
        (candidate for candidate in candidates if candidate.exists()),
        candidates[0],
    )
    if not checkpoint_root.is_dir():
        return checkpoint_root

    candidate_dirs = []
    if (checkpoint_root / "params").exists() or (checkpoint_root / "assets").exists():
        candidate_dirs.append(checkpoint_root)
    candidate_dirs.extend(
        child
        for child in sorted(checkpoint_root.iterdir())
        if child.is_dir() and ((child / "params").exists() or (child / "assets").exists())
    )
    if not candidate_dirs:
        return checkpoint_root

    checkpoint_num = model_cfg.get("checkpoint_num")
    desired_step = _extract_step_number(checkpoint_num)
    if desired_step is not None:
        normalized = str(desired_step)
        for candidate in candidate_dirs:
            name = candidate.name.lstrip("0") or "0"
            if name == normalized:
                return candidate

        for candidate in candidate_dirs:
            candidate_step = _extract_step_number(candidate.name)
            if candidate_step is None:
                continue
            scaled_step = desired_step
            while len(str(scaled_step)) < len(str(candidate_step)):
                scaled_step *= 10
            if candidate_step in {desired_step, scaled_step}:
                return candidate

    numeric_dirs = [
        candidate
        for candidate in candidate_dirs
        if _extract_step_number(candidate.name) is not None
    ]
    if numeric_dirs:
        return max(numeric_dirs, key=lambda candidate: _extract_step_number(candidate.name) or -1)
    return candidate_dirs[0]


def _resolve_train_config(model_cfg: dict[str, Any]):
    config = _config.get_config(model_cfg.get("train_config_name", "pi05_aloha"))
    data_updates: dict[str, Any] = {}
    for key in ("repo_id", "use_delta_joint_actions", "adapt_to_pi"):
        if key in model_cfg:
            data_updates[key] = model_cfg[key]

    norm_stats_path = model_cfg.get("norm_stats_path")
    if norm_stats_path is not None:
        stats_dir = Path(str(norm_stats_path)).expanduser().resolve()
        data_updates["assets"] = _config.AssetsConfig(
            assets_dir=str(stats_dir.parent),
            asset_id=stats_dir.name,
        )

    model_updates: dict[str, Any] = {}
    if "action_horizon" in model_cfg:
        action_horizon = int(model_cfg["action_horizon"])
        if action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {action_horizon}")
        model_updates["action_horizon"] = action_horizon

    replacements: dict[str, Any] = {}
    if data_updates:
        replacements["data"] = dataclasses.replace(config.data, **data_updates)
    if model_updates:
        replacements["model"] = dataclasses.replace(config.model, **model_updates)
    return dataclasses.replace(config, **replacements) if replacements else config


class Model(ModelTemplate):
    def __new__(cls, model_cfg: dict[str, Any] | None = None):
        # Explicit opt-in keeps all existing ALOHA/YAM configurations unchanged.
        if (
            model_cfg is not None
            and model_cfg.get("observation_profile") == "tianji_taccap_pi05_zero_pose"
        ):
            from .pass_ball_model import PassBallZeroPoseModel

            return PassBallZeroPoseModel(model_cfg)
        if (
            model_cfg is not None
            and model_cfg.get("observation_profile") == "tianji_taccap_pi05_pack_plate"
        ):
            from .pack_plate_model import PackPlateZeroPoseModel

            return PackPlateZeroPoseModel(model_cfg)
        return super().__new__(cls)

    def __init__(self, model_cfg: dict[str, Any]):
        self.task_name = model_cfg["task_name"]
        self.inference_seed = model_cfg.get("inference_seed", 0)
        if (
            isinstance(self.inference_seed, bool)
            or not isinstance(self.inference_seed, int)
            or not 0 <= self.inference_seed < 2**32
        ):
            raise ValueError("inference_seed must be an integer in [0, 2**32)")
        self.action_type = model_cfg.get("action_type", "joint")
        env_cfg_type = model_cfg.get("env_cfg_type")
        self.robot_action_dim_info = (
            get_robot_action_dim_info(model_cfg)
            if "robot_action_dim_info" in model_cfg or env_cfg_type is not None else None
        )
        self.robot_action_dim = (
            sum(self.robot_action_dim_info["arm_dim"])
            + sum(self.robot_action_dim_info["ee_dim"])
            if self.robot_action_dim_info is not None
            else None
        )
        self.observation_window: dict[str, Any] | None = None
        self._latest_env_idx_list: list[int] = [0]

        self._train_config = _resolve_train_config(model_cfg)
        expected_profile = (
            "yam_native"
            if type(self._train_config.data).__name__
            in {"LeRobotYamDataConfig", "LeRobotYamJointEeDataConfig"}
            else "aloha"
        )
        self.observation_profile = model_cfg.get("observation_profile", expected_profile)
        if self.observation_profile != expected_profile:
            raise ValueError(
                f"train_config_name={self._train_config.name!r} requires "
                f"observation_profile={expected_profile!r}, got {self.observation_profile!r}"
            )
        self.action_horizon = int(self._train_config.model.action_horizon)
        self.action_dim = int(self._train_config.model.action_dim)
        self.num_steps = int(model_cfg.get("num_steps", 10))
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {self.num_steps}")

        self.policy = self.get_model(model_cfg=model_cfg)
        self.policy.reset_rng(self.inference_seed)
        self.model = self.policy
        warmup_sizes = model_cfg.get("warmup_batch_sizes", [])
        if warmup_sizes:
            if self.observation_profile != "aloha" or self.robot_action_dim is None:
                raise ValueError("batch warmup requires an explicit ALOHA robot configuration")
            observation = {
                "state": np.zeros(self.robot_action_dim, dtype=np.float32),
                "images": {name: np.zeros((3, 224, 224), dtype=np.uint8) for name in
                           ("cam_high", "cam_left_wrist", "cam_right_wrist")},
                "instruction": "Warm up the policy.",
            }
            self.warmup([observation], warmup_sizes)

    def get_model(self, model_cfg: dict[str, Any]):
        repo_id = model_cfg.get("repo_id", "1118")
        model_root = _resolve_pi05_model_root(model_cfg)
        self.model_root = model_root

        norm_stats = None
        norm_stats_path = model_cfg.get("norm_stats_path")
        if norm_stats_path is not None:
            self.norm_stats_path = Path(str(norm_stats_path)).expanduser().resolve()
            norm_stats = _normalize.load(self.norm_stats_path)
        elif repo_id is not None:
            self.norm_stats_path = model_root / "assets" / str(repo_id)
            norm_stats = _normalize.load(self.norm_stats_path)
        else:
            self.norm_stats_path = None

        return _policy_config.create_trained_policy(
            self._train_config,
            str(model_root),
            norm_stats=norm_stats,
        )

    def sampling_modes(self) -> list[str]:
        """Advertise hooks supported by the loaded backend and denoising budget."""
        if self.policy._is_pytorch_model:
            return ["default"]
        modes = ["default", "rtc", "aac", "paint"]
        if self.num_steps >= 3:
            modes.append("autohorizon")
        return modes

    def runtime_metadata(self) -> dict[str, Any]:
        is_pytorch = getattr(self.policy, "_is_pytorch_model", None)
        backend = None if is_pytorch is None else ("pytorch" if is_pytorch else "jax")
        execution: dict[str, Any] = {
            "backend": backend,
            "framework_versions": {},
            "parameter_dtypes": [],
            "parameter_devices": [],
            "compute_dtype": None,
            "noise_dtype": None,
            "inference_seed": self.inference_seed,
            "unavailable": {
                "compute_dtype": "not_measured; parameter_dtype_is_not_end_to_end_precision",
                "noise_dtype": "sampled_noise_is_not_retained; metadata_does_not_run_sampling",
            },
        }
        packages = (
            ("torch", "numpy", "openpi")
            if is_pytorch else ("jax", "jaxlib", "flax", "numpy", "openpi")
        )
        for name in packages:
            loaded = sys.modules.get(name)
            value = getattr(loaded, "__version__", None)
            if value is None:
                try:
                    value = package_metadata.version(name)
                except package_metadata.PackageNotFoundError:
                    execution["unavailable"][f"framework_versions.{name}"] = (
                        "loaded_module_version_and_distribution_metadata_unavailable"
                    )
            execution["framework_versions"][name] = None if value is None else str(value)
        if backend is None:
            execution["unavailable"]["backend"] = "loaded_policy_backend_flag_unavailable"
        else:
            try:
                native_model = self.policy._model
                if is_pytorch:
                    parameters = native_model.parameters()
                else:
                    # Inspect array descriptors only. Never transfer parameter values
                    # to the host or initialize another execution backend for metadata.
                    jax_module = sys.modules["jax"]
                    nnx_module = sys.modules["flax.nnx"]
                    parameters = jax_module.tree.leaves(
                        nnx_module.state(native_model, nnx_module.Param)
                    )
                dtypes = set()
                devices = {}
                for parameter in parameters:
                    dtype = getattr(parameter, "dtype", None)
                    if dtype is not None:
                        dtypes.add(str(dtype))
                    if is_pytorch:
                        device = parameter.device
                        devices[str(device)] = {
                            "platform": str(device.type),
                            "id": device.index,
                        }
                    else:
                        for device in parameter.devices():
                            devices[str(device)] = {
                                "platform": str(device.platform),
                                "id": int(device.id),
                                "device_kind": str(device.device_kind),
                                "process_index": int(device.process_index),
                            }
                execution["parameter_dtypes"] = sorted(dtypes)
                execution["parameter_devices"] = [devices[key] for key in sorted(devices)]
                if not dtypes:
                    execution["unavailable"]["parameter_dtypes"] = "no_parameter_dtype_metadata"
                if not devices:
                    execution["unavailable"]["parameter_devices"] = "no_parameter_device_metadata"
            except Exception as exc:
                execution["unavailable"]["parameter_metadata"] = type(exc).__name__
        execution["rng"] = (
            "policy_owned_torch_generator" if is_pytorch else "policy_owned_jax_key"
        ) if backend is not None else None
        return {
            "policy_family": "pi05",
            "task_name": self.task_name,
            "action_type": self.action_type,
            "train_config_name": self._train_config.name,
            "model_root": str(self.model_root.resolve()),
            "norm_stats_path": (
                None if self.norm_stats_path is None else str(self.norm_stats_path.resolve())
            ),
            "action_horizon": self.action_horizon,
            "action_dim": self.action_dim,
            "num_steps": self.num_steps,
            "inference_seed": self.inference_seed,
            "observation_profile": self.observation_profile,
            "execution": execution,
        }

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        env_ids = [
            obs.get("env_idx", index) for index, obs in enumerate(obs_list)
        ]
        if not env_ids or len(set(env_ids)) != len(env_ids):
            raise ValueError("observations must have nonempty, distinct environment IDs")
        encoded_obs_list = [
            encode_obs(
                obs,
                self.action_type,
                self.robot_action_dim_info,
                observation_profile=self.observation_profile,
            )
            for obs in obs_list
        ]
        self.observation_window = stack_obs(encoded_obs_list)
        self._latest_env_idx_list = env_ids

    def get_action(self, **kwargs):
        kwargs.setdefault("num_steps", self.num_steps)
        action_list = self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]], **kwargs)
        return action_list[0]

    def get_action_rtc(self, sampling: dict[str, Any]):
        required = {"action_condition", "condition_weights", "beta"}
        missing = sorted(required - set(sampling))
        if missing:
            raise ValueError(f"RTC sampling is missing fields: {missing}")

        condition = np.array(
            sampling["action_condition"], dtype=np.float32, copy=True
        )
        weights = np.array(
            sampling["condition_weights"], dtype=np.float32, copy=True
        )
        beta = float(sampling["beta"])
        expected_action_dim = self.robot_action_dim or self.action_dim
        if condition.shape != (self.action_horizon, expected_action_dim):
            raise ValueError(
                "action_condition must have shape "
                f"{(self.action_horizon, expected_action_dim)}, got {condition.shape}"
            )
        if weights.shape != (self.action_horizon,):
            raise ValueError(
                f"condition_weights must have shape {(self.action_horizon,)}, got {weights.shape}"
            )
        if not np.isfinite(condition).all() or not np.isfinite(weights).all():
            raise ValueError("RTC sampling arrays must be finite")
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError(f"RTC beta must be finite and positive, got {beta}")

        return self.get_action(
            action_condition=condition,
            condition_weights=weights,
            rtc_beta=beta,
        )

    def get_action_paint(self, sampling: dict[str, Any]):
        required = {"action_prefix", "delay_steps"}
        missing = sorted(required - set(sampling))
        if missing:
            raise ValueError(f"PAINT sampling is missing fields: {missing}")

        delay_steps = int(sampling["delay_steps"])
        expected_action_dim = self.robot_action_dim or self.action_dim
        prefix = np.asarray(sampling["action_prefix"], dtype=np.float32)
        if not 0 < delay_steps < self.action_horizon:
            raise ValueError(
                "PAINT delay_steps must satisfy "
                f"0 < d < {self.action_horizon}, got {delay_steps}"
            )
        if prefix.shape != (delay_steps, expected_action_dim):
            raise ValueError(
                "PAINT action_prefix must have shape "
                f"{(delay_steps, expected_action_dim)}, got {prefix.shape}"
            )
        if not np.isfinite(prefix).all():
            raise ValueError("PAINT action_prefix must be finite")

        condition = np.zeros(
            (self.action_horizon, expected_action_dim),
            dtype=np.float32,
        )
        condition[:delay_steps] = prefix
        actions = self.get_action(
            paint_action_condition=condition,
            paint_delay_steps=delay_steps,
        )
        return {
            "actions": actions,
            "paint": {
                "delay_steps": delay_steps,
                "num_steps": self.num_steps,
                "model_evaluations": 3 * self.num_steps,
                "inversion": "backward_euler",
            },
        }

    def get_action_aac(self, sampling: dict[str, Any]):
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")
        if len(self._latest_env_idx_list) != 1:
            raise ValueError("AAC sampling requires exactly one observation")

        num_samples = int(sampling.get("num_samples", 20))
        if num_samples <= 1:
            raise ValueError("AAC num_samples must be greater than one")

        single_observation = slice_stacked_obs(self.observation_window, 0)
        actions = np.asarray(
            self.policy.infer(
                single_observation,
                num_steps=self.num_steps,
                num_samples=num_samples,
            )["actions"]
        )
        expected_action_dim = self.robot_action_dim or self.action_dim
        expected_shape = (num_samples, self.action_horizon, expected_action_dim)
        if actions.shape != expected_shape:
            raise ValueError(
                f"Pi05 AAC actions must have shape {expected_shape}, got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("Pi05 AAC actions must be finite")

        if self.robot_action_dim_info is None:
            candidates = [sample for sample in actions]
        else:
            candidates = [
                unpack_robot_state(
                    sample,
                    self.action_type,
                    self.robot_action_dim_info,
                    source_type="obs",
                )
                for sample in actions
            ]
        return {"actions": candidates}

    def get_action_autohorizon(self, sampling: dict[str, Any]):
        if set(sampling) - {"mode"}:
            raise ValueError("AutoHorizon does not accept sampling overrides")
        if self.num_steps < 3:
            raise ValueError("AutoHorizon requires at least three denoising steps")
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")
        if len(self._latest_env_idx_list) != 1:
            raise ValueError("AutoHorizon sampling requires exactly one observation")

        single_observation = slice_stacked_obs(self.observation_window, 0)
        result = self.policy.infer(
            single_observation,
            num_steps=self.num_steps,
            autohorizon=True,
        )
        actions = np.asarray(result["actions"])
        expected_action_dim = self.robot_action_dim or self.action_dim
        expected_shape = (self.action_horizon, expected_action_dim)
        if actions.shape != expected_shape:
            raise ValueError(
                f"Pi05 AutoHorizon actions must have shape {expected_shape}, got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("Pi05 AutoHorizon actions must be finite")

        metadata = result.get("autohorizon")
        if not isinstance(metadata, dict):
            raise ValueError("Pi05 AutoHorizon inference did not return metadata")
        execution_steps = metadata.get("execution_steps")
        if (
            not isinstance(execution_steps, int)
            or isinstance(execution_steps, bool)
            or not 1 <= execution_steps <= self.action_horizon
        ):
            raise ValueError(
                "Pi05 AutoHorizon execution_steps must satisfy "
                f"1 <= e <= {self.action_horizon}, got {execution_steps!r}"
            )

        if self.robot_action_dim_info is None:
            decoded = actions
        else:
            decoded = unpack_robot_state(
                actions,
                self.action_type,
                self.robot_action_dim_info,
                source_type="obs",
            )
        return {"actions": decoded, "autohorizon": metadata}

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")

        env_idx_list = list(self._latest_env_idx_list if env_idx_list is None else env_idx_list)
        if len(set(env_idx_list)) != len(env_idx_list):
            raise ValueError("requested environment IDs must be distinct")
        if not env_idx_list:
            return []
        positions = {env_id: i for i, env_id in enumerate(self._latest_env_idx_list)}
        missing = set(env_idx_list) - positions.keys()
        if missing:
            raise ValueError(f"no observation for environment IDs: {sorted(missing)}")
        observations = [slice_stacked_obs(self.observation_window, positions[i]) for i in env_idx_list]
        kwargs.setdefault("num_steps", self.num_steps)
        noise = kwargs.get("noise")
        if noise is not None:
            noise = np.asarray(noise)
            if len(observations) == 1 and noise.shape == (self.action_horizon, self.action_dim):
                noise = noise[None]
            if noise.shape != (len(observations), self.action_horizon, self.action_dim):
                raise ValueError("noise must contain one [horizon, action_dim] row per requested environment")
            kwargs["noise"] = noise if len(observations) > 1 else noise[0]
        if len(observations) == 1:
            # Retain the single-observation API, including conditioned sampling.
            batched_actions = [self.policy.infer(observations[0], **kwargs)["actions"]]
        else:
            if (kwargs.get("num_samples", 1) != 1 or kwargs.get("autohorizon", False)
                    or kwargs.get("action_condition") is not None
                    or kwargs.get("paint_action_condition") is not None):
                raise ValueError("multi-environment inference supports default sampling only")
            batched_actions = np.asarray(self.policy.infer(stack_obs(observations), **kwargs)["actions"])
            if batched_actions.ndim != 3 or batched_actions.shape[:2] != (len(observations), self.action_horizon):
                raise ValueError(f"invalid batched action shape: {batched_actions.shape}")
        action_list = []
        for actions in batched_actions:
            if not np.isfinite(actions).all():
                raise ValueError("policy returned non-finite actions")
            if self.robot_action_dim_info is None:
                action_list.append(actions)
            else:
                action_list.append(
                    unpack_robot_state(
                        actions,
                        self.action_type,
                        self.robot_action_dim_info,
                        source_type="obs",
                    )
                )

        return action_list

    def warmup(self, obs_list, batch_sizes):
        """Compile requested batch shapes without advancing the evaluation RNG."""
        sizes = list(batch_sizes)
        if not sizes or any(type(n) is not int or not 1 <= n <= 4 for n in sizes):
            raise ValueError("warmup batch sizes must be integers in [1, 4]")
        old_window, old_ids = self.observation_window, self._latest_env_idx_list
        pytorch = self.policy._is_pytorch_model
        rng = self.policy._torch_generator.get_state() if pytorch else self.policy._rng
        try:
            for size in sorted(set(sizes)):
                self.update_obs_batch([{**obs_list[i % len(obs_list)], "env_idx": i} for i in range(size)])
                self.get_action_batch(noise=np.zeros((size, self.action_horizon, self.action_dim), dtype=np.float32))
        finally:
            if pytorch:
                self.policy._torch_generator.set_state(rng)
            else:
                self.policy._rng = rng
            self.observation_window, self._latest_env_idx_list = old_window, old_ids

    def reset(self):
        self.policy.reset_rng(self.inference_seed)
        self.observation_window = None
        self._latest_env_idx_list = [0]

    def reset_obsrvationwindows(self):
        self.reset()


def encode_obs(
    observation,
    action_type,
    robot_action_dim_info,
    *,
    observation_profile="aloha",
):
    if observation_profile == "yam_native":
        return encode_yam_obs(observation, action_type, robot_action_dim_info)
    if observation_profile != "aloha":
        raise ValueError(f"Unsupported Pi05 observation profile: {observation_profile!r}")

    if "images" in observation and "state" in observation:
        state = np.asarray(observation["state"], dtype=np.float32)
        images = {
            "cam_high": ensure_chw_uint8(observation["images"]["cam_high"]),
            "cam_left_wrist": ensure_chw_uint8(observation["images"]["cam_left_wrist"]),
            "cam_right_wrist": ensure_chw_uint8(observation["images"]["cam_right_wrist"]),
        }
        prompt = observation.get("instruction")
        return {"state": state, "images": images, "prompt": prompt}

    if robot_action_dim_info is None:
        raise ValueError("env_cfg_type is required when encoding raw environment observations.")

    images = {
        "cam_high": ensure_chw_uint8(extract_image(observation, ["cam_high", "cam_head", "head_camera", "top_camera"])),
        "cam_left_wrist": ensure_chw_uint8(
            extract_image(observation, ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"])
        ),
        "cam_right_wrist": ensure_chw_uint8(
            extract_image(observation, ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"])
        ),
    }
    state = pack_robot_state(observation, action_type, robot_action_dim_info, source_type="obs").astype(np.float32)
    prompt = observation.get("instruction")
    return {"state": state, "images": images, "prompt": prompt}


def encode_yam_obs(observation, action_type, robot_action_dim_info):
    if robot_action_dim_info is None:
        raise ValueError("env_cfg_type is required when encoding YAM observations.")

    if "observation/state" in observation:
        state = np.asarray(observation["observation/state"], dtype=np.float32)
    elif "state" in observation and not isinstance(observation["state"], dict):
        state = np.asarray(observation["state"], dtype=np.float32)
    else:
        state = pack_robot_state(
            observation,
            action_type,
            robot_action_dim_info,
            source_type="obs",
        ).astype(np.float32)

    def raw_image(key, candidates):
        if key in observation:
            return np.asarray(observation[key])
        return np.asarray(extract_image(observation, candidates))

    return {
        "observation/image": raw_image(
            "observation/image",
            ["cam_high", "cam_head", "head_camera", "top_camera"],
        ),
        "observation/left_wrist": raw_image(
            "observation/left_wrist",
            ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"],
        ),
        "observation/right_wrist": raw_image(
            "observation/right_wrist",
            ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"],
        ),
        "observation/state": state,
        "prompt": observation.get("instruction", observation.get("prompt")),
    }


def stack_obs(obs_list: list[dict[str, Any]]) -> dict[str, Any]:
    if "observation/state" in obs_list[0]:
        return {
            "observation/state": np.stack(
                [obs["observation/state"] for obs in obs_list], axis=0
            ),
            "observation/image": np.stack(
                [obs["observation/image"] for obs in obs_list], axis=0
            ),
            "observation/left_wrist": np.stack(
                [obs["observation/left_wrist"] for obs in obs_list], axis=0
            ),
            "observation/right_wrist": np.stack(
                [obs["observation/right_wrist"] for obs in obs_list], axis=0
            ),
            "prompt": [obs["prompt"] for obs in obs_list],
        }
    return {
        "state": np.stack([obs["state"] for obs in obs_list], axis=0),
        "images": {
            "cam_high": np.stack([obs["images"]["cam_high"] for obs in obs_list], axis=0),
            "cam_left_wrist": np.stack([obs["images"]["cam_left_wrist"] for obs in obs_list], axis=0),
            "cam_right_wrist": np.stack([obs["images"]["cam_right_wrist"] for obs in obs_list], axis=0),
        },
        "prompt": [obs["prompt"] for obs in obs_list],
    }


def slice_stacked_obs(obs: dict[str, Any], batch_index: int) -> dict[str, Any]:
    if "observation/state" in obs:
        return {
            "observation/state": obs["observation/state"][batch_index],
            "observation/image": obs["observation/image"][batch_index],
            "observation/left_wrist": obs["observation/left_wrist"][batch_index],
            "observation/right_wrist": obs["observation/right_wrist"][batch_index],
            "prompt": obs["prompt"][batch_index],
        }
    return {
        "state": obs["state"][batch_index],
        "images": {
            "cam_high": obs["images"]["cam_high"][batch_index],
            "cam_left_wrist": obs["images"]["cam_left_wrist"][batch_index],
            "cam_right_wrist": obs["images"]["cam_right_wrist"][batch_index],
        },
        "prompt": obs["prompt"][batch_index],
    }


def extract_image(observation, candidate_names):
    vision = observation.get("vision", observation.get("images", {}))
    for candidate_name in candidate_names:
        if candidate_name not in vision:
            continue
        image = vision[candidate_name]
        if isinstance(image, dict):
            for image_key in ("color", "rgb"):
                if image_key in image:
                    return image[image_key]
        else:
            return image
    raise KeyError(f"Could not find any image for candidates: {candidate_names}")


def ensure_chw_uint8(image):
    image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(f"Expected image ndim=3, got shape {image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if image.shape[-1] in (1, 3):
        image_hwc = image
    elif image.shape[0] in (1, 3):
        image_hwc = np.transpose(image, (1, 2, 0))
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    return np.transpose(image_hwc, (2, 0, 1))
