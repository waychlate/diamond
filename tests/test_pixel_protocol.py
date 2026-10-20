import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_pixel_data import prepare
from evaluate_pixels import error_totals, transition_batches, with_mae
from cross_validate_pixels import aggregate_if_complete, training_command
from pixel_mask import build_mask_from_fullres
from data import Dataset, Episode


class PixelProtocolTests(unittest.TestCase):
    def test_mask_uses_target_vehicles_and_empty_masks_are_undefined(self):
        target = torch.full((1, 3, 48, 320), -1.0)
        target[:, 2, 20:28, 140:155] = 1  # Blue traffic.
        target[:, 1, 20:28, 160:175] = 1  # Green ego.
        mask = build_mask_from_fullres(target, 48, 320)
        self.assertGreater(mask.sum().item(), 0)
        self.assertEqual(mask[0, 0, 0, 0].item(), 0)
        prediction = target.clone()
        prediction[:, :, :10, :10] += 0.5  # Only the background is wrong.
        metrics = with_mae(error_totals(prediction, target, mask))
        self.assertGreater(metrics["unmasked_mae"], 0)
        self.assertEqual(metrics["masked_mae"], 0)
        empty = with_mae(error_totals(prediction, target, torch.zeros_like(mask)))
        self.assertIsNone(empty["masked_mae"])
        self.assertEqual(empty["empty_mask_transitions"], 1)

    def test_one_step_uses_real_context_final_frame_and_stops_at_terminal(self):
        obs = torch.arange(6).view(6, 1, 1, 1).float()
        ep = Episode(obs, torch.arange(6), torch.zeros(6), torch.zeros(6, dtype=torch.uint8),
                     torch.zeros(6, dtype=torch.uint8), {"final_observation": torch.full((1, 1, 1), 6.0)})
        batches = list(transition_batches(ep, 4, 2))
        self.assertEqual([i for batch in batches for i in batch[0]], [4, 5, 6])
        self.assertEqual(batches[0][1][1].flatten().tolist(), [1, 2, 3, 4])
        self.assertEqual(batches[-1][3].item(), 6)
        ep.end[4] = 1
        self.assertEqual([i for batch in transition_batches(ep, 4, 2) for i in batch[0]], [4, 5])

    def test_preparation_preserves_holdout_and_reuses_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw"
            for split, count in (("train", 10), ("test", 2)):
                (source / split).mkdir(parents=True)
                for i in range(count):
                    name = source / split / f"episode_{i:04d}"
                    np.savez_compressed(str(name) + "_visuals.npz", visuals=np.zeros((7, 150, 600, 3), dtype=np.uint8))
                    with Path(str(name) + "_data.csv").open("w") as handle:
                        writer = csv.writer(handle)
                        writer.writerow(["action", "reward", "done", "crashed"])
                        writer.writerows([[0, 0, False, 0]] * 7)
            output = root / "prepared"
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(source, output)
                prepare(source, output)  # Repeated preparation is read-only.
            seen = []
            for fold in range(5):
                path = output / "folds" / f"fold_{fold}"
                split = json.loads((path / "split.json").read_text())
                self.assertFalse(set(split["train_ids"]) & set(split["validation_ids"]))
                self.assertEqual(len(split["train_ids"]), 8)
                self.assertEqual(len(split["validation_ids"]), 2)
                seen.extend(split["validation_ids"])
                for local in ("train", "test"):
                    ds = Dataset(path / local)
                    ds.load_from_default_path()
                    self.assertEqual(len(ds.load_episode(0)), 6)
                    link = ds._get_episode_path(0)
                    self.assertTrue(link.is_symlink())
                    self.assertTrue(link.resolve().is_relative_to(output / "train"))
            self.assertEqual(sorted(seen), list(range(10)))
            holdout = Dataset(output / "test")
            holdout.load_from_default_path()
            self.assertEqual(holdout.num_episodes, 2)
            with self.assertRaises(ValueError):
                prepare(source, output, seed=1)

    def test_cv_starts_fresh_and_aggregates_all_folds(self):
        command = training_command(Path("/data"), Path("/run"), 2)
        self.assertIn("initialization.path_to_ckpt=null", command)
        self.assertIn("agent.denoiser.sigma_data=0.15", command)
        self.assertIn("training.num_final_epochs=50", command)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            aggregate_if_complete(root)
            self.assertFalse((root / "cv_summary.json").exists())
            for fold in range(5):
                path = root / f"fold_{fold}" / "pixel_eval"
                path.mkdir(parents=True)
                row = {"unmasked_mae": fold + 1, "masked_mae": (fold + 1) / 2,
                       "absolute_error": fold + 1, "pixel_channels": 1,
                       "masked_absolute_error": (fold + 1) / 2, "mask_weight_channels": 1}
                (path / "summary.json").write_text(json.dumps(row))
            with contextlib.redirect_stdout(io.StringIO()):
                aggregate_if_complete(root)
            result = json.loads((root / "cv_summary.json").read_text())
            self.assertEqual(result["pooled_unmasked_mae"], 3)
            self.assertEqual(result["masked_mae_fold_mean"], 1.5)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
