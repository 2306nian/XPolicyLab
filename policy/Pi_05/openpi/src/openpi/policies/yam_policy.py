"""Policy transforms for bimanual YAM LeRobot datasets.

Dataset layout:
  observation.state              (14,) = [L: 6 joints + gripper, R: 6 joints + gripper]
  action                         (14,) with the same ordering
  observation.ee_pose            optional (24,) = per arm [position 3 + rotation matrix 9]
  action.ee_pose                 optional (24,) with the same ordering
  observation.images.top_rgb     third-person camera
  observation.images.left_rgb    left-side camera
  observation.images.right_rgb   right-side camera
"""

import dataclasses

import einops
import numpy as np
from scipy.spatial.transform import Rotation

from openpi import transforms
from openpi.models import model as _model

YAM_ACTION_DIM = 14
YAM_EE_POSE_DIM = 24
YAM_JOINT_EE_ACTION_DIM = 26


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 4 and image.shape[1] == 3:
        image = einops.rearrange(image, "b c h w -> b h w c")
    elif image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class YamInputs(transforms.DataTransformFn):
    """Map YAM observations/actions onto OpenPI's canonical model inputs."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        mask = np.ones(np.asarray(data["observation/state"]).shape[:-1], dtype=bool)
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": _parse_image(data["observation/image"]),
                "left_wrist_0_rgb": _parse_image(data["observation/left_wrist"]),
                "right_wrist_0_rgb": _parse_image(data["observation/right_wrist"]),
            },
            "image_mask": {
                "base_0_rgb": mask.copy(),
                "left_wrist_0_rgb": mask.copy(),
                "right_wrist_0_rgb": mask.copy(),
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


def relative_ee_actions(current_pose: np.ndarray, target_poses: np.ndarray) -> np.ndarray:
    """Convert absolute dual-arm EE targets into current-EE-frame 6D deltas."""
    current_pose = np.asarray(current_pose, dtype=np.float32)
    target_poses = np.asarray(target_poses, dtype=np.float32)
    if current_pose.shape != (YAM_EE_POSE_DIM,):
        raise ValueError(f"expected current EE pose ({YAM_EE_POSE_DIM},), got {current_pose.shape}")
    if target_poses.shape[-1] != YAM_EE_POSE_DIM:
        raise ValueError(
            f"expected target EE pose last dim {YAM_EE_POSE_DIM}, got {target_poses.shape}"
        )

    deltas = []
    for offset in (0, 12):
        current_position = current_pose[offset : offset + 3]
        current_rotation = current_pose[offset + 3 : offset + 12].reshape(3, 3)
        target_position = target_poses[..., offset : offset + 3]
        target_rotation = target_poses[..., offset + 3 : offset + 12].reshape(
            *target_poses.shape[:-1], 3, 3
        )
        relative_position = np.einsum(
            "ij,...j->...i", current_rotation.T, target_position - current_position
        )
        relative_rotation = np.einsum("ij,...jk->...ik", current_rotation.T, target_rotation)
        relative_axis_angle = Rotation.from_matrix(
            relative_rotation.reshape(-1, 3, 3)
        ).as_rotvec().reshape(*target_poses.shape[:-1], 3)
        deltas.append(np.concatenate([relative_position, relative_axis_angle], axis=-1))
    return np.concatenate(deltas, axis=-1).astype(np.float32)


@dataclasses.dataclass(frozen=True)
class YamJointEeInputs(transforms.DataTransformFn):
    """Build joint actions plus 12D dual-arm EE auxiliary targets."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        inputs = YamInputs(model_type=self.model_type)(data)
        has_ee_observation = "observation/ee_pose" in data
        has_ee_action = "action_ee_pose" in data
        if "actions" in data and has_ee_observation and has_ee_action:
            ee_actions = relative_ee_actions(
                data["observation/ee_pose"], data["action_ee_pose"]
            )
            inputs["actions"] = np.concatenate(
                [np.asarray(data["actions"], dtype=np.float32), ee_actions], axis=-1
            )
        elif has_ee_observation != has_ee_action:
            raise ValueError("joint+EE training requires both observation.ee_pose and action.ee_pose")
        return inputs


@dataclasses.dataclass(frozen=True)
class YamOutputs(transforms.DataTransformFn):
    """Remove OpenPI action padding while preserving YAM joint/gripper order."""

    action_dim: int = YAM_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}
