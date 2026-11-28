import copy
import unittest
from unittest.mock import patch

import torch
from torch.nn.functional import cross_entropy

from loopsight import build_loopsight
from train_loopsight_arc import parse_args as parse_training_args
from ttt_loopsight_arc import parse_args as parse_ttt_args


def inputs(device, size=8):
    pixels = torch.arange(2 * size * size, device=device).reshape(2, size, size) % 12
    tasks = torch.tensor([0, 1], device=device)
    mask = torch.ones_like(pixels, dtype=torch.bool)
    mask[:, : size // 2, size // 2 :] = False
    return pixels, tasks, mask


def model_for_test(scale, device, **kwargs):
    torch.manual_seed(42)
    model = build_loopsight(scale, num_tasks=2, image_size=8, **kwargs).to(device)
    # Exercise the modulation and write-gate paths with nonzero learned projections.
    with torch.no_grad():
        for layer in model.core_layers:
            layer.time_modulation[-1].weight.normal_(std=0.02)
        model.fusion_gate.up.weight.normal_(std=0.08)
    return model


class OptimizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_cpu_preserves_outputs_and_checkpoint_keys(self):
        model = model_for_test("small", "cpu").eval()
        optimized = copy.deepcopy(model).enable_optimization()
        optimized.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(list(model.state_dict()), list(optimized.state_dict()))
        with torch.inference_mode():
            expected, _ = model(*inputs("cpu"), override_max_steps=2)
            actual, metadata = optimized(*inputs("cpu"), override_max_steps=2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(metadata.max_steps, 2)

    def test_cli_opt_in_preserves_the_amp_setting(self):
        for parse_args in (parse_training_args, parse_ttt_args):
            with self.subTest(parser=parse_args.__module__):
                default = parse_args([])
                self.assertFalse(default.optimize)
                self.assertFalse(default.no_amp)
                optimized = parse_args(["--optimize"])
                self.assertTrue(optimized.optimize)
                self.assertFalse(optimized.no_amp)
                self.assertTrue(parse_args(["--optimize", "--no-amp"]).no_amp)
                self.assertTrue(parse_args(["--fp32-optimization"]).optimize)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.cuda and torch.cuda.get_device_capability()[0] >= 8,
    "requires an Ampere or newer CUDA GPU",
)
class CudaOptimizationTest(unittest.TestCase):
    def test_inference_matches_eager_at_all_scales(self):
        from loopsight import _inference_fp32

        pixels, tasks, mask = inputs("cuda")
        with torch.inference_mode():
            for scale in ("small", "medium", "large"):
                with self.subTest(scale=scale):
                    eager = model_for_test(scale, "cuda").eval()
                    optimized = copy.deepcopy(eager).enable_optimization()
                    with patch.object(
                        _inference_fp32, "forward_final_fp32", wraps=_inference_fp32.forward_final_fp32,
                    ) as fast:
                        for padding in (None, mask):
                            expected, expected_meta = eager(pixels, tasks, padding, override_max_steps=3)
                            actual, meta = optimized(pixels, tasks, padding, override_max_steps=3)
                            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=5e-4)
                            torch.testing.assert_close(meta.exit_steps, expected_meta.exit_steps)
                            self.assertEqual(meta.max_steps, expected_meta.max_steps)
                        self.assertEqual(fast.call_count, 2)
                        self.assertEqual(fast.call_args.kwargs["overlap_memory"], scale != "large")

                        expected, _, _ = eager(pixels, tasks, mask, override_max_steps=3, return_intermediates=True)
                        actual, _, _ = optimized(
                            pixels, tasks, mask, override_max_steps=3, return_intermediates=True,
                        )
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                        self.assertEqual(fast.call_count, 2)

    def test_training_matches_gradients_at_two_loop_depths(self):
        from loopsight import _training_fusion

        eager = model_for_test("small", "cuda", loop_drop_path_rate=0.1).train()
        optimized = copy.deepcopy(eager).enable_optimization()
        pixels, tasks, mask = inputs("cuda")
        for steps in (2, 3):
            with self.subTest(steps=steps):
                with patch.object(
                    _training_fusion, "residual_conditioned_rms", wraps=_training_fusion.residual_conditioned_rms,
                ) as fused:
                    for model in (eager, optimized):
                        model.zero_grad(set_to_none=True)
                        torch.manual_seed(120)
                        logits, _ = model(pixels, tasks, mask, override_max_steps=steps)
                        cross_entropy(logits, pixels).backward()
                    self.assertEqual(fused.call_count, steps * optimized.loop_core_depth)

                for (name, reference), (_, actual) in zip(
                    eager.named_parameters(), optimized.named_parameters(), strict=True,
                ):
                    self.assertIsNotNone(reference.grad, name)
                    self.assertIsNotNone(actual.grad, name)
                    torch.testing.assert_close(actual.grad, reference.grad, atol=5e-6, rtol=2e-3)

    def test_training_fusion_preserves_default_cuda_amp(self):
        from loopsight import _training_fusion

        eager = model_for_test("small", "cuda", loop_drop_path_rate=0.1).train()
        optimized = copy.deepcopy(eager).enable_optimization()
        pixels, tasks, mask = inputs("cuda")
        steps = 3
        outcomes = []
        with patch.object(
            _training_fusion, "residual_conditioned_rms",
            wraps=_training_fusion.residual_conditioned_rms,
        ) as fused:
            for model in (eager, optimized):
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                scaler = torch.amp.GradScaler("cuda", init_scale=128.0)
                self.assertTrue(scaler.is_enabled())
                torch.manual_seed(120)
                with torch.autocast("cuda"):
                    logits, _ = model(pixels, tasks, mask, override_max_steps=steps)
                    self.assertEqual(logits.dtype, torch.get_autocast_dtype("cuda"))
                    loss = cross_entropy(logits, pixels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                outcomes.append((logits.detach(), loss.detach()))
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all().item())

                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                self.assertGreaterEqual(scaler.get_scale(), scale_before)

            self.assertEqual(fused.call_count, steps * optimized.loop_core_depth)

        torch.testing.assert_close(outcomes[0][0], outcomes[1][0], atol=3e-3, rtol=5e-3)
        torch.testing.assert_close(outcomes[0][1], outcomes[1][1], atol=1e-3, rtol=2e-3)
        # AMP can move individual values across FP16 rounding boundaries after fusion.
        # Compare the full gradient norm and the scaled optimizer update.
        reference = torch.cat([param.grad.flatten() for param in eager.parameters()])
        actual = torch.cat([param.grad.flatten() for param in optimized.parameters()])
        relative_error = torch.linalg.vector_norm(actual - reference) / torch.linalg.vector_norm(reference)
        self.assertLess(relative_error.item(), 2e-3)
        for expected, updated in zip(eager.parameters(), optimized.parameters(), strict=True):
            torch.testing.assert_close(updated, expected, atol=1e-5, rtol=1e-3)
