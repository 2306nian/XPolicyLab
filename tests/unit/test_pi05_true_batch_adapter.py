"""不加载权重，直接执行 Pi05 适配器方法验证真实 batch 契约。"""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "policy/Pi_05/model.py"


def make_adapter(policy):
    tree = ast.parse(SOURCE.read_text())
    model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Model")
    model.bases = []
    methods = {"update_obs", "update_obs_batch", "get_action", "get_action_batch", "reset", "warmup"}
    model.body = [n for n in model.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    helpers = {"stack_obs", "slice_stacked_obs", "encode_obs", "encode_yam_obs", "ensure_chw_uint8", "extract_image"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in helpers]
    namespace = {"np": np, "Any": Any}
    exec(compile(ast.Module(body=[*nodes, model], type_ignores=[]), str(SOURCE), "exec"), namespace)
    adapter = namespace["Model"]()
    adapter.observation_window = None
    adapter._latest_env_idx_list = [0]
    adapter.action_type, adapter.observation_profile = "joint", "aloha"
    adapter.robot_action_dim_info = None
    adapter.action_horizon, adapter.action_dim, adapter.num_steps = 3, 4, 10
    adapter.inference_seed = 0
    adapter.policy = policy
    return adapter


def observation(env_id):
    return {"env_idx": env_id, "state": np.full(4, env_id, dtype=np.float32),
            "images": {k: np.full((3, 8, 8), env_id, dtype=np.uint8)
                       for k in ("cam_high", "cam_left_wrist", "cam_right_wrist")},
            "instruction": f"task {env_id}"}


class Pi05TrueBatchTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.policy = SimpleNamespace(infer=self.infer, _is_pytorch_model=False, _rng=17)
        self.adapter = make_adapter(self.policy)
        self.adapter.update_obs_batch([observation(i) for i in (2, 5, 9, 12)])

    def infer(self, obs, **kwargs):
        self.calls.append((obs, kwargs))
        self.policy._rng += 1
        actions = np.repeat(obs["state"][..., None, :], 3, axis=-2)
        return {"actions": actions + kwargs.get("noise", 0)}

    def test_reordered_partial_batches_each_call_policy_once(self):
        for ids in ([12, 2, 9, 5], [9, 2, 5], [5, 12], [9]):
            with self.subTest(ids=ids):
                self.calls.clear()
                outputs = self.adapter.get_action_batch(ids)
                self.assertEqual(len(self.calls), 1)
                for env_id, output in zip(ids, outputs):
                    np.testing.assert_array_equal(output, np.full((3, 4), env_id))
                obs, kwargs = self.calls[0]
                self.assertEqual(kwargs["num_steps"], 10)
                if len(ids) > 1:
                    self.assertEqual(obs["prompt"], [f"task {i}" for i in ids])
                    self.assertEqual(obs["images"]["cam_high"].shape, (len(ids), 3, 8, 8))

    def test_fixed_per_env_noise_matches_independent_calls(self):
        noise = np.random.default_rng(0).standard_normal((4, 3, 4)).astype(np.float32)
        batched = self.adapter.get_action_batch(noise=noise)
        singles = [self.adapter.get_action_batch([env_id], noise=noise[i])[0]
                   for i, env_id in enumerate((2, 5, 9, 12))]
        np.testing.assert_array_equal(batched, singles)

    def test_mapping_and_noise_errors_fail_before_inference(self):
        for ids in ([2, 2], [3]):
            with self.assertRaises(ValueError):
                self.adapter.get_action_batch(ids)
        with self.assertRaises(ValueError):
            self.adapter.get_action_batch(noise=np.zeros((3, 4)))
        with self.assertRaises(ValueError):
            self.adapter.update_obs_batch([observation(2), observation(2)])
        self.assertEqual(self.adapter.get_action_batch([]), [])
        self.assertEqual(self.calls, [])

    def test_conditioned_sampling_is_not_silently_changed_to_default_batch(self):
        for kwargs in ({"num_samples": 2}, {"autohorizon": True},
                       {"action_condition": np.zeros((3, 4))},
                       {"paint_action_condition": np.zeros((3, 4))}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.adapter.get_action_batch(**kwargs)
        self.assertEqual(self.calls, [])

    def test_bad_outputs_do_not_fall_back_to_serial_calls(self):
        self.policy.infer = lambda *a, **k: {"actions": np.zeros((3, 4))}
        with self.assertRaisesRegex(ValueError, "shape"):
            self.adapter.get_action_batch()
        self.policy.infer = lambda *a, **k: {"actions": np.full((4, 3, 4), np.nan)}
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.adapter.get_action_batch()

    def test_warmup_restores_rng_and_observation_even_on_failure(self):
        window = self.adapter.observation_window
        self.adapter.warmup([observation(0)], [1, 2, 3, 4])
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(self.policy._rng, 17)
        self.assertIs(self.adapter.observation_window, window)
        self.assertEqual(self.adapter._latest_env_idx_list, [2, 5, 9, 12])

        def broken(*args, **kwargs):
            self.policy._rng = 99
            raise RuntimeError("device failure")

        self.policy.infer = broken
        with self.assertRaises(RuntimeError):
            self.adapter.warmup([observation(0)], [4])
        self.assertEqual(self.policy._rng, 17)
        self.assertIs(self.adapter.observation_window, window)

    def test_yam_images_and_masks_preserve_batch_axis_including_three(self):
        try:
            import einops
        except ImportError:
            self.skipTest("YAM transform requires einops")
        path = ROOT / "policy/Pi_05/openpi/src/openpi/policies/yam_policy.py"
        tree = ast.parse(path.read_text())
        parse_image = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_parse_image")
        inputs = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "YamInputs")
        call = next(n for n in inputs.body if isinstance(n, ast.FunctionDef) and n.name == "__call__")
        namespace = {"np": np, "einops": einops}
        exec(compile(ast.Module(body=[parse_image, call], type_ignores=[]), str(path), "exec"), namespace)
        for size in (1, 2, 3, 4):
            images = np.arange(size * 3 * 8 * 8).reshape(size, 3, 8, 8).astype(np.uint8)
            expected = images.transpose(0, 2, 3, 1)
            for value in (images, expected):
                data = {"observation/state": np.zeros((size, 14)),
                        **{k: value for k in ("observation/image", "observation/left_wrist", "observation/right_wrist")}}
                result = namespace["__call__"](None, data)
                np.testing.assert_array_equal(result["image"]["base_0_rgb"], expected)
                np.testing.assert_array_equal(result["image_mask"]["base_0_rgb"], np.ones(size, dtype=bool))


if __name__ == "__main__":
    unittest.main()
