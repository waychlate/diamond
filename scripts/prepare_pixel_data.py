"""Convert raw episodes once and make five disjoint, symlink-backed CV datasets.

Folds follow KineMamba's src/make_cv_folds.py: train and test episodes pooled in sorted
file order, split with sklearn KFold(5, shuffle=True, random_state=seed). The same
assignment is written to cv_folds.json in KineMamba's format so both models can read it.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import Dataset
from convert_and_process import convert_data


def source_manifest(source):
    manifest = {}
    for split in ("train", "test"):
        csvs = sorted((source / split).glob("episode_*_data.csv"))
        if not csvs:
            raise ValueError(f"No episode CSV files in {source / split}")
        visuals = sorted(p.name.removesuffix("_visuals.npz") for p in (source / split).glob("*_visuals.npz"))
        if visuals != [p.name.removesuffix("_data.csv") for p in csvs]:
            # KineMamba lists episodes by their visuals files; the two listings must agree.
            raise ValueError(f"Episode CSV and visuals files differ in {source / split}")
        records = []
        for csv in csvs:
            images = csv.with_name(csv.name.replace("_data.csv", "_visuals.npz"))
            for path in (csv, images):
                stat = path.stat()  # Missing pairs must fail rather than silently disappear.
                records.append((path.name, stat.st_size, stat.st_mtime_ns))
        manifest[split] = {"episodes": [p.stem.removesuffix("_data") for p in csvs],
                           "fingerprint": hashlib.sha256(json.dumps(records).encode()).hexdigest()}
    return manifest


def kfold(n, n_folds=5, seed=0):
    """Validation indices of sklearn KFold(n_folds, shuffle=True, random_state=seed), without sklearn."""
    order = np.arange(n)
    np.random.RandomState(seed).shuffle(order)
    sizes = np.full(n_folds, n // n_folds)
    sizes[: n % n_folds] += 1
    return [np.sort(order[stop - size:stop]) for size, stop in zip(sizes, np.cumsum(sizes))]


def make_subset(episodes, destination):
    """episodes: (converted dataset, episode id) pairs, possibly from both splits."""
    subset = Dataset(destination, save_on_disk=False)
    # Metadata comes from these episodes only; links reference the one converted copy.
    for source, original_id in episodes:
        episode = source.load_episode(int(original_id))
        new_id = subset.add_episode(episode)
        link = subset._get_episode_path(new_id)
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(source._get_episode_path(int(original_id)).resolve())
    subset.save_to_default_path()


def prepare(source, output, seed=0):
    source, output = Path(source).resolve(), Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"version": 2, "source": str(source), "seed": seed,
                "raw": source_manifest(source)}
    # An array job and the meeting evaluation can safely request the same preparation.
    with output.with_name(output.name + ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ready = output / "manifest.json"
        if output.exists():
            if not ready.exists() or json.loads(ready.read_text()) != manifest:
                raise ValueError(f"{output} is incomplete or belongs to different data/seed; choose a new output directory")
            return output
        with tempfile.TemporaryDirectory(prefix=output.name + ".", dir=output.parent) as tmp:
            stage = Path(tmp) / "data"
            convert_data(source, stage)
            datasets = {}
            for split in ("train", "test"):
                ds = Dataset(stage / split)
                ds.load_from_default_path()
                if ds.num_episodes != len(manifest["raw"][split]["episodes"]):
                    raise ValueError(f"Conversion skipped episodes in {split}; preparation aborted")
                datasets[split] = ds
            # Pooled order matches KineMamba: train episodes, then test, each sorted by name.
            pooled = [(split, local_id) for split in ("train", "test")
                      for local_id in range(datasets[split].num_episodes)]
            names = [f"{split}/{manifest['raw'][split]['episodes'][i]}_visuals.npz" for split, i in pooled]
            if len(pooled) < 5:
                raise ValueError("Five-fold CV requires at least five episodes")
            folds = []
            for fold, val_ids in enumerate(kfold(len(pooled), 5, seed)):
                train_ids = np.setdiff1d(np.arange(len(pooled)), val_ids)
                folds.append({"train": [names[i] for i in train_ids], "val": [names[i] for i in val_ids]})
                fold_dir = stage / "folds" / f"fold_{fold}"
                make_subset([(datasets[pooled[i][0]], pooled[i][1]) for i in train_ids], fold_dir / "train")
                make_subset([(datasets[pooled[i][0]], pooled[i][1]) for i in val_ids], fold_dir / "test")
                (fold_dir / "split.json").write_text(json.dumps({
                    "fold": fold, "seed": seed, "train_ids": train_ids.tolist(),
                    "validation_ids": val_ids.tolist(),
                    "train_episodes": folds[-1]["train"], "validation_episodes": folds[-1]["val"],
                    "note": "fold test/ is validation; ids index the pooled train+test episode list",
                }, indent=2) + "\n")
            (stage / "cv_folds.json").write_text(json.dumps({
                "seed": seed, "n_folds": 5, "data_dir": str(source),
                "n_episodes": len(pooled), "folds": folds}, indent=2) + "\n")
            # Links must point at the final location, not the temporary staging directory.
            for link in (stage / "folds").rglob("*.pt"):
                if link.is_symlink():
                    relative = link.resolve().relative_to(stage)
                    link.unlink()
                    link.symlink_to(output / relative)
            (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            stage.rename(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(prepare(args.source, args.output, args.seed))
