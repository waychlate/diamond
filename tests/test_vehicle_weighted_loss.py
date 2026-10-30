from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from data import Batch
from models.diffusion.denoiser import Denoiser, DenoiserConfig, vehicle_weighted_pixel_mae
from vehicle_mask import build_mask_from_fullres
from cross_validate_pixels import run_fold, training_command


class ConstantModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.noise_emb = nn.Embedding(1, 1)
        self.bias = nn.Parameter(torch.tensor(0.2))

    def forward(self, noisy_obs, noise_level, obs, act):
        return self.bias.expand_as(noisy_obs)


def tiny_denoiser(pixel_weight, mse_weight=1.0):
    # Exercise the real denoising forward pass with a tiny substitute for the UNet.
    denoiser = Denoiser.__new__(Denoiser)
    nn.Module.__init__(denoiser)
    denoiser.cfg = DenoiserConfig(SimpleNamespace(num_steps_conditioning=1), 0.15, 0.0,
                                  pixel_mask_gain=20.0, pixel_mae_weight=pixel_weight,
                                  diffusion_mse_weight=mse_weight)
    denoiser.inner_model = ConstantModel()
    denoiser.sample_sigma_training = lambda n, device: torch.ones(n, device=device)
    return denoiser


class VehicleWeightedLossTests(unittest.TestCase):
    def test_weight_normalization_and_larger_vehicle_gradients(self):
        target = torch.full((1, 3, 8, 16), -1.0)
        target[:, 2, 2:6, 4:8] = 1.0
        prediction = (target + 0.25).requires_grad_()
        mask = build_mask_from_fullres(target, 8, 16)
        gain = 20.0
        weights = 1 + gain * mask
        actual = vehicle_weighted_pixel_mae(prediction, target, gain)
        # Constant error must stay constant regardless of RGB count or mask mass.
        self.assertAlmostEqual(actual.item(), 0.25, places=6)
        actual.backward()
        self.assertTrue(torch.allclose(prediction.grad, weights.expand_as(target) / (3 * weights.sum())))
        self.assertGreater(prediction.grad[0, 0, 3, 5], prediction.grad[0, 0, 0, 0])

    def test_no_vehicles_or_zero_gain_recovers_plain_mae(self):
        target = torch.zeros(2, 3, 8, 16)
        prediction = torch.rand_like(target)
        self.assertTrue(torch.allclose(vehicle_weighted_pixel_mae(prediction, target, 20),
                                       (prediction - target).abs().mean()))
        target[:, 2, 2:6, 4:8] = 1.0
        self.assertTrue(torch.allclose(vehicle_weighted_pixel_mae(prediction, target, 0),
                                       (prediction - target).abs().mean()))

    def test_pixel_term_backpropagates_and_padding_is_excluded(self):
        obs = torch.zeros(2, 3, 3, 8, 16)
        obs[0, :, 2, 2:6, 4:8] = 1.0
        obs[1] = 999  # Completely padded sample must not affect either loss.
        batch = Batch(obs, torch.zeros(2, 3, dtype=torch.long), None, None, None,
                      torch.tensor([[True, True, True], [False, False, False]]), [], [])
        baseline = tiny_denoiser(0)
        weighted = tiny_denoiser(1)
        torch.manual_seed(7)
        baseline_loss, baseline_logs = baseline(batch)
        baseline_loss.backward()
        torch.manual_seed(7)
        weighted_loss, logs = weighted(batch)
        weighted_loss.backward()
        self.assertTrue(torch.allclose(baseline_loss.detach(), baseline_logs["loss_diffusion_mse"]))
        self.assertTrue(torch.allclose(logs["loss_diffusion_mse"], baseline_loss.detach()))
        self.assertTrue(torch.allclose(weighted_loss.detach(),
                                       logs["loss_diffusion_mse"] + logs["loss_weighted_pixel_mae"]))
        self.assertNotEqual(baseline.inner_model.bias.grad.item(), weighted.inner_model.bias.grad.item())
        self.assertTrue(torch.isfinite(weighted.inner_model.bias.grad))

    def test_pixel_mae_only_drops_the_diffusion_term(self):
        obs = torch.zeros(1, 2, 3, 8, 16)
        obs[0, 1, 2, 2:6, 4:8] = 1.0
        batch = Batch(obs, torch.zeros(1, 2, dtype=torch.long), None, None, None,
                      torch.ones(1, 2, dtype=torch.bool), [], [])
        torch.manual_seed(7)
        loss, logs = tiny_denoiser(1, mse_weight=0)(batch)
        self.assertTrue(torch.allclose(loss.detach(), logs["loss_weighted_pixel_mae"]))
        self.assertGreater(logs["loss_diffusion_mse"].item(), 0)  # Still logged.

    def test_cv_forwards_loss_settings_and_rejects_changed_resume(self):
        command = training_command(Path("/data"), Path("/run"), 0,
                                   pixel_mask_gain=20, pixel_mae_weight=1, diffusion_mse_weight=0)
        self.assertIn("agent.denoiser.pixel_mask_gain=20", command)
        self.assertIn("agent.denoiser.pixel_mae_weight=1", command)
        self.assertIn("agent.denoiser.diffusion_mse_weight=0", command)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data/folds/fold_0").mkdir(parents=True)
            (root / "data/folds/fold_0/split.json").write_text("{}")
            (root / "runs/fold_0/checkpoints").mkdir(parents=True)
            (root / "runs/fold_0/checkpoints/state.pt").touch()
            (root / "runs/fold_0/config").mkdir()
            (root / "runs/fold_0/config/trainer.yaml").write_text("agent:\n  denoiser: {}\n")
            args = SimpleNamespace(data=root / "data", output=root / "runs", fold=0,
                                   resume=True, pixel_mask_gain=20, pixel_mae_weight=1,
                                   diffusion_mse_weight=0)
            with patch("cross_validate_pixels.subprocess.run") as run:
                with self.assertRaisesRegex(ValueError, "Cannot resume with changed"):
                    run_fold(args)
                run.assert_not_called()

    def test_fully_padded_targets_produce_finite_zero_loss(self):
        model = tiny_denoiser(1)
        batch = Batch(torch.zeros(1, 2, 3, 8, 16), torch.zeros(1, 2, dtype=torch.long),
                      None, None, None, torch.zeros(1, 2, dtype=torch.bool), [], [])
        loss, logs = model(batch)
        self.assertEqual(loss.item(), 0)
        self.assertEqual(logs["loss_weighted_pixel_mae"].item(), 0)
        loss.backward()

    def test_padded_target_steps_do_not_dilute_loss_or_gradients(self):
        for pixel_weight in (0, 1):
            model = tiny_denoiser(pixel_weight)
            results = []
            for length in (2, 3):
                obs = torch.zeros(1, length, 3, 8, 16)
                obs[:, :, 2, 2:6, 4:8] = 1
                valid = torch.zeros(1, length, dtype=torch.bool)
                valid[:, :2] = True
                batch = Batch(obs, torch.zeros(1, length, dtype=torch.long),
                              None, None, None, valid, [], [])
                model.zero_grad()
                torch.manual_seed(7)
                loss, logs = model(batch)
                loss.backward()
                results.append((loss.detach(), logs, model.inner_model.bias.grad.clone()))
            self.assertTrue(torch.allclose(results[0][0], results[1][0]))
            self.assertTrue(torch.allclose(results[0][2], results[1][2]))
            for metric in results[0][1]:
                self.assertTrue(torch.allclose(results[0][1][metric], results[1][1][metric]))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
