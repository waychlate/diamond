from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import Dataset, Episode
from trainer import Trainer


def make_trainer(root, static):
    # Exercise production checkpoint methods without constructing a GPU training run.
    trainer = Trainer.__new__(Trainer)
    trainer._rank = 0
    trainer._is_static_dataset = static
    trainer._device = torch.device("cpu")
    trainer._path_state_ckpt = root / "state.pt"
    trainer._keep_agent_copies = lambda weights, epoch: None
    trainer._save_info_for_import_script = lambda epoch: None
    trainer.epoch = 3
    trainer.agent = torch.nn.Linear(2, 1)
    for name in ("train_dataset", "test_dataset"):
        dataset = Dataset(root / name)
        dataset.add_episode(Episode(torch.zeros(5, 3, 8, 16), torch.zeros(5, dtype=torch.long),
                                    torch.zeros(5), torch.zeros(5, dtype=torch.uint8),
                                    torch.zeros(5, dtype=torch.uint8), {}))
        dataset.save_to_default_path()
        setattr(trainer, name, dataset)
    return trainer


class TrainerCheckpointTests(unittest.TestCase):
    def test_static_checkpoint_never_writes_shared_dataset_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            trainer = make_trainer(Path(temporary), static=True)
            before = {ds._default_path: (ds._default_path.read_bytes(), ds._default_path.stat().st_mtime_ns)
                      for ds in (trainer.train_dataset, trainer.test_dataset)}
            with patch.object(Dataset, "save_to_default_path", side_effect=AssertionError("Shared dataset write")):
                trainer.save_checkpoint()
            self.assertTrue(trainer._path_state_ckpt.is_file())
            for path, (content, modified) in before.items():
                self.assertEqual(path.read_bytes(), content)
                self.assertEqual(path.stat().st_mtime_ns, modified)

    def test_online_checkpoint_still_persists_dataset_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            trainer = make_trainer(Path(temporary), static=False)
            trainer.train_dataset.counter_rew[1] += 2
            trainer.save_checkpoint()
            loaded = Dataset(trainer.train_dataset._directory)
            loaded.load_from_default_path()
            self.assertEqual(loaded.counter_rew[1], 2)

    def test_full_training_checkpoint_resumes_numpy_metadata_and_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            trainer = make_trainer(Path(temporary), static=True)
            trainer.save_checkpoint()
            expected_weights = {key: value.clone() for key, value in trainer.agent.state_dict().items()}
            trainer.epoch = 0
            with torch.no_grad():
                trainer.agent.weight.zero_()
            trainer.train_dataset.num_steps = 0
            trainer.load_state_checkpoint()
            self.assertEqual(trainer.epoch, 3)
            self.assertEqual(trainer.train_dataset.num_steps, 5)
            self.assertEqual(trainer.train_dataset.lengths.tolist(), [5])
            for key, value in trainer.agent.state_dict().items():
                self.assertTrue(torch.equal(value, expected_weights[key]))


if __name__ == "__main__":
    unittest.main()
