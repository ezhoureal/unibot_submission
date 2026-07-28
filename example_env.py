"""Reference evaluation env: builds stub observations from a policy's metadata
and validates the actions it returns. Used by run_client.py, not run directly."""

from __future__ import annotations

import hmac
import os
from typing import Iterable, Mapping

import numpy as np

# Per-frame (shape, dtype) for each observation key.
OBSERVATION_SPEC: dict[str, tuple[tuple[int, ...], np.dtype]] = {
    "observation.images.cam_left_high":   ((480, 640, 3), np.dtype("uint8")),
    "observation.images.cam_right_high":  ((480, 640, 3), np.dtype("uint8")),
    "observation.images.cam_left_wrist":  ((480, 640, 3), np.dtype("uint8")),
    "observation.images.cam_right_wrist": ((480, 640, 3), np.dtype("uint8")),
    "observation.state.left_arm":         ((7,),          np.dtype("float32")),
    "observation.state.right_arm":        ((7,),          np.dtype("float32")),
    "observation.state.left_ee_pose_gripper_base":  ((6,), np.dtype("float32")),
    "observation.state.right_ee_pose_gripper_base": ((6,), np.dtype("float32")),
    "observation.state.left_gripper":     ((1,),          np.dtype("float32")),
    "observation.state.right_gripper":    ((1,),          np.dtype("float32")),
    "observation.state.lower_body":       ((15,),         np.dtype("float32")),
    "observation.language":               (None,           np.dtype("str")),
}

# Per-frame shape of each action key, one dict per control space.
JOINT_ACTION_SPEC = {
    "action.left_arm":  (7,),
    "action.right_arm": (7,),
    "action.left_gripper":  (1,),
    "action.right_gripper": (1,),
    "action.pivot":         (7,),
    "meta.token":            None,
}

EE_ACTION_SPEC = {
    "action.left_ee_pose_gripper_base":  (6,),
    "action.right_ee_pose_gripper_base": (6,),
    "action.left_gripper":  (1,),
    "action.right_gripper": (1,),
    "action.pivot":         (7,),
    "meta.token":            None,
}

CONTROL_SPACES = ("joint", "ee")
DEFAULT_LANGUAGE = "move the block to the target position."  # placeholder task instruction


class ActionError(ValueError):
    """Raised when a policy's `get_action` response violates the spec."""


class ExampleEnv:
    """Validates a policy's metadata and actions; serves stub observations."""

    def __init__(self, metadata: dict):
        """Validate the metadata and pick the action spec for its control space."""
        self.policy_metadata = metadata
        self._expected_token = os.environ.get("UNIBOT_SUBMISSION_TOKEN")
        if self._expected_token is None:
            raise ValueError("UNIBOT_SUBMISSION_TOKEN is required")
        self.validate_token(self.policy_metadata["token"])

        self.data_keys = self.policy_metadata["data_keys"]
        self.obs_chunk_size = self.policy_metadata["obs_chunk_size"]
        self.action_chunk_size = self.policy_metadata["action_chunk_size"]
        self.control_space = self.policy_metadata["control_space"]

        for key in self.data_keys:
            if key not in OBSERVATION_SPEC:
                raise ActionError(f"unknown observation key {key!r}; expected one of {OBSERVATION_SPEC.keys()}")
        if len(self.data_keys) == 0:
            raise ActionError("data_keys must be a non-empty list")
        if not (type(self.obs_chunk_size) == int and self.obs_chunk_size > 0):
            raise ActionError(f"obs_chunk_size must be a positive int, got {self.obs_chunk_size!r}")
        if not (type(self.action_chunk_size) == int and self.action_chunk_size > 0):
            raise ActionError(f"action_chunk_size must be a positive int, got {self.action_chunk_size!r}")
        if self.control_space not in CONTROL_SPACES:
            raise ActionError(f"unknown control_space {self.control_space!r}; expected one of {CONTROL_SPACES}")
        self.action_spec = JOINT_ACTION_SPEC if self.control_space == "joint" else EE_ACTION_SPEC

    def validate_token(self, token: str):
        """Reject a token that doesn't match UNIBOT_SUBMISSION_TOKEN."""
        if not hmac.compare_digest(token, self._expected_token):
            raise ActionError("invalid meta.token: does not match UNIBOT_SUBMISSION_TOKEN")

    def reset(self):
        """Start an episode; return the first observation."""
        return self.get_observation()

    def step(self, action: dict):
        """Validate one action; return the next observation."""
        self.validate_token(action["meta.token"])

        for action_key in self.action_spec:
            if action_key not in action:
                raise ActionError(f"missing action key {action_key!r}; expected one of {self.action_spec.keys()}")

        for action_key in action:
            if action_key not in self.action_spec:
                raise ActionError(f"unknown action key {action_key!r}; expected one of {self.action_spec.keys()}")
            if action_key != "meta.token":
                if not isinstance(action[action_key], np.ndarray):
                    raise ActionError(f"action[{action_key}] must be a numpy.ndarray, got {type(action[action_key]).__name__}")
                if action[action_key].shape != (self.action_chunk_size, *self.action_spec[action_key]):
                    raise ActionError(f"action[{action_key}].shape {action[action_key].shape} != expected {(self.action_chunk_size, *self.action_spec[action_key])}")
                if action[action_key].dtype != np.float32:
                    raise ActionError(f"action[{action_key}].dtype {action[action_key].dtype} != np.float32")
        return self.get_observation()

    def get_observation(self):
        """Build a spec-shaped stub observation (zeros; language is a string)."""
        observation = {}
        for key in self.data_keys:
            if key == "observation.language":
                observation[key] = DEFAULT_LANGUAGE
                continue
            frame_shape, dtype = OBSERVATION_SPEC[key]
            shape = (self.obs_chunk_size,) + frame_shape
            observation[key] = np.zeros(shape, dtype=dtype)
        return observation
