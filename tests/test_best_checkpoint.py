import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trainer import Trainer


class BestCheckpointTests(unittest.TestCase):
    def test_best_agent_only_replaced_by_a_lower_test_loss(self):
        with tempfile.TemporaryDirectory() as temporary:
            trainer = SimpleNamespace(_path_ckpt_dir=Path(temporary), epoch=1,
                                      agent=torch.nn.Linear(1, 1))
            log = lambda loss: [{"denoiser/test/loss_denoising": torch.tensor(loss)}, {"other": 1.0}]
            torch.nn.init.constant_(trainer.agent.weight, 1.0)
            Trainer.save_best_agent(trainer, log(0.5))
            torch.nn.init.constant_(trainer.agent.weight, 2.0)
            trainer.epoch = 2
            Trainer.save_best_agent(trainer, log(0.7))  # Worse: keeps epoch 1.
            self.assertEqual(torch.load(Path(temporary) / "best.pt")["weight"].item(), 1.0)
            trainer.epoch = 3
            Trainer.save_best_agent(trainer, log(0.4))  # Better: replaces it.
            self.assertEqual(torch.load(Path(temporary) / "best.pt")["weight"].item(), 2.0)
            Trainer.save_best_agent(trainer, [{"unrelated": 1.0}])  # No denoiser loss: no-op.

    def test_joint_ttc_selects_on_pixel_plus_ttc_like_kinemamba(self):
        with tempfile.TemporaryDirectory() as temporary:
            trainer = SimpleNamespace(_path_ckpt_dir=Path(temporary), epoch=1, agent=torch.nn.Linear(1, 1))

            def log(pixel, ttc, weight=0.1):  # The training total down-weights TTC; selection must not.
                return [{"denoiser/test/loss_weighted_pixel_mae": torch.tensor(pixel),
                         "denoiser/test/loss_ttc": torch.tensor(ttc),
                         "denoiser/test/loss_denoising": torch.tensor(pixel + weight * ttc)}]

            torch.nn.init.constant_(trainer.agent.weight, 1.0)
            Trainer.save_best_agent(trainer, log(0.01, 0.8))   # pixel + ttc = 0.81, total = 0.09
            torch.nn.init.constant_(trainer.agent.weight, 2.0)
            trainer.epoch = 2
            Trainer.save_best_agent(trainer, log(0.001, 0.85))  # 0.851 is worse, although its total 0.086 is lower.
            record = json.loads((Path(temporary) / "best.json").read_text())
            self.assertEqual((record["epoch"], record["metric"]), (1, "weighted_pixel_mae + ttc_mae"))
            self.assertEqual(torch.load(Path(temporary) / "best.pt")["weight"].item(), 1.0)

    def test_record_from_another_metric_is_not_compared(self):
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "best.json").write_text(json.dumps({"epoch": 9, "test_loss": 0.0001}))
            trainer = SimpleNamespace(_path_ckpt_dir=Path(temporary), epoch=1, agent=torch.nn.Linear(1, 1))
            Trainer.save_best_agent(trainer, [{"denoiser/test/loss_weighted_pixel_mae": torch.tensor(0.01),
                                               "denoiser/test/loss_ttc": torch.tensor(0.8)}])
            self.assertEqual(json.loads((Path(temporary) / "best.json").read_text())["epoch"], 1)


    def test_held_out_evaluation_runs_without_autograd(self):
        seen = []

        class Model(torch.nn.Module):
            def forward(self, batch):
                seen.append(torch.is_grad_enabled())
                return torch.tensor(0.0), {}

        counter = SimpleNamespace(get=lambda name: 0, set=lambda name, value: None)
        batch = SimpleNamespace(to=lambda device: batch)
        trainer = SimpleNamespace(agent=SimpleNamespace(denoiser=Model()), _device="cpu", num_batch_test=counter,
                                  _data_loader_test=SimpleNamespace(get=lambda name: [batch]))
        Trainer.test_component(trainer, "denoiser")
        self.assertEqual(seen, [False])


if __name__ == "__main__":
    unittest.main()
