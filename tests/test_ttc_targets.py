from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_latent_ttc import compute_ttc_targets_for_batch, evaluate_test_split


class TtcTargetTests(unittest.TestCase):
    def test_offset_one_targets_the_next_frame_and_its_mask(self):
        seq = 7  # 3 context frames + 4 windows
        mask = torch.tensor([[False, False, True, True, True, True, True]])
        batch = SimpleNamespace(end=torch.zeros(1, seq), mask_padding=mask,
                                info=[{"obs_ttc": torch.arange(seq).float()}])
        now, now_mask = compute_ttc_targets_for_batch(batch, num_cond=3, offset=0)
        nxt, nxt_mask = compute_ttc_targets_for_batch(batch, num_cond=3, offset=1)
        self.assertEqual(now[0, :, 0].tolist(), [2, 3, 4, 5])
        self.assertEqual(nxt[0, :, 0].tolist(), [3, 4, 5, 6])
        self.assertEqual(now_mask[0].tolist(), mask[0, 2:6].tolist())
        self.assertEqual(nxt_mask[0].tolist(), mask[0, 3:7].tolist())

    def test_full_evaluation_scores_every_window_once(self):
        frames = torch.arange(10).float().view(10, 1, 1, 1).expand(10, 3, 2, 2)
        episode = SimpleNamespace(obs=frames, act=torch.zeros(10, dtype=torch.long),
                                  info={"obs_ttc": torch.arange(10).float() + 0.5})
        dataset = SimpleNamespace(num_episodes=2, load_episode=lambda i: episode)
        agent = SimpleNamespace(denoiser=SimpleNamespace(
            extract_latent=lambda obs, act: obs[:, -1].mean(dim=(1, 2, 3)).view(-1, 1, 1, 1)))

        class Head(torch.nn.Module):
            def forward(self, latents):
                return latents.flatten(1)[:, :1], None  # Predicts the last context frame's index.

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "ttc"
            summary = evaluate_test_split(agent, Head(), dataset, num_cond=4, offset=1, output=output,
                                          info={}, device=torch.device("cpu"), chunk=4)
            # Windows start at 0..5 (target frames 4..9 would need ttc[10]); the target is 1.5 above each prediction.
            self.assertEqual(summary["transitions"], 12)
            self.assertAlmostEqual(summary["ttc_mae"], 1.5, places=6)
            self.assertEqual(json.loads((output / "summary.json").read_text())["episodes"], 2)


if __name__ == "__main__":
    unittest.main()
