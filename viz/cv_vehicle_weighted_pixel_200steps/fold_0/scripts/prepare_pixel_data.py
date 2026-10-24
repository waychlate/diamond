"""Convert raw episodes once and make five disjoint, symlink-backed CV datasets."""
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
        records = []
        for csv in csvs:
            images = csv.with_name(csv.name.replace("_data.csv", "_visuals.npz"))
            for path in (csv, images):
                stat = path.stat()  # Missing pairs must fail rather than silently disappear.
                records.append((path.name, stat.st_size, stat.st_mtime_ns))
        manifest[split] = {"episodes": [p.stem.removesuffix("_data") for p in csvs],
                           "fingerprint": hashlib.sha256(json.dumps(records).encode()).hexdigest()}
    return manifest


def make_subset(source, ids, destination):
    subset = Dataset(destination, save_on_disk=False)
    # Metadata comes from these episodes only; links reference the one converted copy.
    for original_id in ids:
        episode = source.load_episode(int(original_id))
        new_id = subset.add_episode(episode)
        link = subset._get_episode_path(new_id)
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(source._get_episode_path(int(original_id)).resolve())
    subset.save_to_default_path()


def prepare(source, output, seed=0):
    source, output = Path(source).resolve(), Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"version": 1, "source": str(source), "seed": seed,
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
            train = datasets["train"]
            if train.num_episodes < 5:
                raise ValueError("Five-fold CV requires at least five training episodes")
            chunks = np.array_split(np.random.default_rng(seed).permutation(train.num_episodes), 5)
            for fold, held_out in enumerate(chunks):
                train_ids = np.sort(np.concatenate([part for i, part in enumerate(chunks) if i != fold]))
                val_ids = np.sort(held_out)
                fold_dir = stage / "folds" / f"fold_{fold}"
                make_subset(train, train_ids, fold_dir / "train")
                make_subset(train, val_ids, fold_dir / "test")
                (fold_dir / "split.json").write_text(json.dumps({
                    "fold": fold, "seed": seed, "train_ids": train_ids.tolist(),
                    "validation_ids": val_ids.tolist(),
                    "train_episodes": [manifest["raw"]["train"]["episodes"][i] for i in train_ids],
                    "validation_episodes": [manifest["raw"]["train"]["episodes"][i] for i in val_ids],
                    "note": "fold test/ is validation; the preserved test set is never in CV",
                }, indent=2) + "\n")
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
