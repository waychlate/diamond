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


if __name__ == "__main__":
    unittest.main()
