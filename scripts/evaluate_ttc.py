"""Score the TTC head trained jointly inside a DIAMOND run on every target from frame num_cond on."""
import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def main(args):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from agent import Agent
    from data import Dataset
    from train_latent_ttc import evaluate_test_split

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    if float(OmegaConf.select(cfg, "agent.denoiser.ttc_loss_weight", default=0.0)) <= 0:
        raise ValueError("This run did not train a TTC head (agent.denoiser.ttc_loss_weight is 0)")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    agent = Agent(instantiate(cfg.agent, num_actions=cfg.env.num_actions)).to(device).eval()
    agent.load(args.checkpoint, load_rew_end_model=False, load_actor_critic=False)
    dataset = Dataset(Path(args.dataset), "test_dataset")
    dataset.load_from_default_path()
    evaluate_test_split(agent, agent.denoiser.ttc_head, dataset, cfg.agent.denoiser.inner_model.num_steps_conditioning,
                        1, args.output, {"denoiser_checkpoint": str(args.checkpoint), "dataset": str(args.dataset),
                                         "trained_jointly": True, "ttc_loss_weight": float(cfg.agent.denoiser.ttc_loss_weight),
                                         "seed": args.seed}, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Converted split directory, e.g. folds/fold_0/test")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="trainer.yaml of the run")
    parser.add_argument("--output", type=Path, required=True, help="New results directory")
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
