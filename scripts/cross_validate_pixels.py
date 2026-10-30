"""Train one independent CV fold, then evaluate its final model using pixel MAE."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def training_command(dataset, run, fold, *, resume=False, epochs=50, workers=8,
                     pixel_mask_gain=0.0, pixel_mae_weight=0.0, diffusion_mse_weight=1.0):
    command = [sys.executable, str(ROOT / "src/main.py"),
               f"static_dataset.path={dataset}", f"hydra.run.dir={run}",
               "initialization.path_to_ckpt=null", "agent.denoiser.sigma_data=0.15",
               # 20 frames of context, as in KineMamba; 240 lets the action embedding split evenly over 20 frames.
               "agent.denoiser.inner_model.num_steps_conditioning=20",
               "agent.denoiser.inner_model.cond_channels=240",
               f"agent.denoiser.pixel_mask_gain={pixel_mask_gain}",
               f"agent.denoiser.pixel_mae_weight={pixel_mae_weight}",
               f"agent.denoiser.diffusion_mse_weight={diffusion_mse_weight}",
               f"common.seed={fold}", "common.devices=all", "wandb.mode=disabled",
               f"training.num_final_epochs={epochs}", f"training.num_workers_data_loaders={workers}",
               "training.cache_in_ram=False", "training.compile_wm=False"]
    if resume:
        command += ["common.resume=True", "hydra.output_subdir=null"]
    return command


def run_fold(args):
    dataset = args.data.resolve() / "folds" / f"fold_{args.fold}"
    if not (dataset / "split.json").is_file():
        raise FileNotFoundError(f"Prepare the dataset first: {dataset}")
    run = args.output.resolve() / f"fold_{args.fold}"
    if run.exists() and not args.resume:
        raise FileExistsError(f"{run} already exists; use --resume or a new --output")
    if args.resume and not (run / "checkpoints/state.pt").is_file():
        raise FileNotFoundError(f"Cannot resume without {run / 'checkpoints/state.pt'}")
    if args.resume:
        from omegaconf import OmegaConf
        previous = OmegaConf.load(run / "config/trainer.yaml")
        for name, default in (("pixel_mask_gain", 0.0), ("pixel_mae_weight", 0.0), ("diffusion_mse_weight", 1.0)):
            saved = OmegaConf.select(previous, f"agent.denoiser.{name}", default=default)
            if float(saved) != getattr(args, name):
                raise ValueError(f"Cannot resume with changed {name}: saved {saved}, requested {getattr(args, name)}. "
                                 "Use the saved loss settings or start a new output directory.")
    subprocess.run(training_command(dataset, run, args.fold, resume=args.resume,
                                   epochs=args.epochs, workers=args.workers,
                                   pixel_mask_gain=args.pixel_mask_gain,
                                   pixel_mae_weight=args.pixel_mae_weight,
                                   diffusion_mse_weight=args.diffusion_mse_weight), cwd=ROOT, check=True)
    evaluation = run / "pixel_eval"
    if (evaluation / "summary.json").is_file():
        print(f"Fold evaluation already complete: {evaluation}")
        return
    if evaluation.exists():
        raise FileExistsError(f"Incomplete evaluation at {evaluation}; move it aside before retrying")
    subprocess.run([sys.executable, str(ROOT / "scripts/evaluate_pixels.py"),
                    "--dataset", str(dataset / "test"),
                    "--checkpoint", str(run / "checkpoints/state.pt"),
                    "--config", str(run / "config/trainer.yaml"),
                    "--checkpoint-training-sigma", "0.15", "--sigma-data", "0.15",
                    "--seed", "0", "--batch-size", str(args.batch_size),
                    "--output", str(evaluation)], cwd=ROOT, check=True)


def aggregate(output):
    import numpy as np
    output = Path(output).resolve()
    summaries = []
    for fold in range(5):
        path = output / f"fold_{fold}" / "pixel_eval/summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"All five folds must finish before aggregation; missing {path}")
        summaries.append(json.loads(path.read_text()))
    for setting in ("training_pixel_mask_gain", "training_pixel_mae_weight", "training_diffusion_mse_weight"):
        if len({summary.get(setting) for summary in summaries}) != 1:
            raise ValueError(f"Cannot aggregate folds trained with different {setting}")
    result = {"folds": 5, "fold_results": summaries,
              "note": "CV validation episodes; folds pool the train and test splits as in KineMamba"}
    for metric in ("masked_mae", "unmasked_mae", "masked_mae_kinemamba_scale", "unmasked_mae_kinemamba_scale"):
        values = [summary[metric] for summary in summaries]
        if any(value is None for value in values):
            raise ValueError(f"A fold has undefined {metric}; inspect empty masks")
        result[metric + "_fold_mean"] = float(np.mean(values))
        result[metric + "_fold_std"] = float(np.std(values, ddof=1))
    result["pooled_unmasked_mae"] = sum(s["absolute_error"] for s in summaries) / sum(s["pixel_channels"] for s in summaries)
    result["pooled_masked_mae"] = sum(s["masked_absolute_error"] for s in summaries) / sum(s["mask_weight_channels"] for s in summaries)
    (output / "cv_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "fold_results"}, indent=2))


def aggregate_if_complete(output):
    output = Path(output).resolve()
    if all((output / f"fold_{fold}/pixel_eval/summary.json").is_file() for fold in range(5)):
        with (output / "aggregate.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            aggregate(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pixel-mask-gain", type=float, default=0.0)
    parser.add_argument("--pixel-mae-weight", type=float, default=0.0)
    parser.add_argument("--diffusion-mse-weight", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.fold is None and "SLURM_ARRAY_TASK_ID" in os.environ:
        try:
            args.fold = int(os.environ["SLURM_ARRAY_TASK_ID"])
        except ValueError:
            parser.error("SLURM_ARRAY_TASK_ID must be an integer from 0 to 4")
        if args.fold not in range(5):
            parser.error("SLURM_ARRAY_TASK_ID must be an integer from 0 to 4")
    if args.aggregate:
        aggregate(args.output)
    elif args.fold is None or args.data is None:
        parser.error("--data and --fold are required (Slurm array jobs supply --fold automatically)")
    else:
        run_fold(args)
        aggregate_if_complete(args.output)
