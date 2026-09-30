"""Dataset helper.

  python -m src.prepare_dataset check --task fruit
  python -m src.prepare_dataset split --source raw_images --task defect [--mapping map.json]
  python -m src.prepare_dataset split --source raw/train --test-source raw/test --task defect --mapping m.json
"""
import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

from .config import DATASET_DIR, SEED, SPLITS, TASKS
from .preprocessing import DatasetError, images_in, validate_dataset


def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(task):
    class_names = validate_dataset(task)
    print(f"Structure OK for '{task}'. Classes: {class_names}\n")
    seen, corrupt, leaks = {}, [], []
    for split in SPLITS:
        counts = {c: len(images_in(DATASET_DIR / task / split / c)) for c in class_names}
        print(f"{split:<11}{counts}")
        for p in images_in(DATASET_DIR / task / split):
            try:
                with Image.open(p) as im:
                    im.verify()
            except Exception:
                corrupt.append(p)
                continue
            h = file_hash(p)
            if h in seen and seen[h][0] != split:
                leaks.append((seen[h][1], p))
            seen.setdefault(h, (split, p))
    ok = True
    if corrupt:
        ok = False
        print(f"\nCorrupt images ({len(corrupt)}) - delete them:")
        for p in corrupt[:20]:
            print("  ", p)
    if leaks:
        ok = False
        print(f"\nDATA LEAKAGE: {len(leaks)} identical images appear in different splits:")
        for a, b in leaks[:20]:
            print(f"   {a}  ==  {b}")
    if ok:
        print("\nNo corrupt files and no exact duplicates across splits.")
    return ok


def _pool(source, mapping, label):
    source = Path(source)
    if not source.is_dir():
        raise DatasetError(f"{label} folder not found: {source}")
    pooled = defaultdict(list)
    for folder in sorted(p for p in source.iterdir() if p.is_dir()):
        if mapping is not None:
            if folder.name not in mapping:
                print(f"[{label}] skipping '{folder.name}' (not in mapping)")
                continue
            target = mapping[folder.name]
        else:
            target = folder.name
        pooled[target.lower()] += [(folder.name, f) for f in images_in(folder)]
    return pooled


def _copy(files, dest):
    dest.mkdir(parents=True, exist_ok=True)
    for folder_name, f in files:
        shutil.copy2(f, dest / f"{folder_name}_{f.name}")


def split_dataset(source, task, mapping=None, val_ratio=0.15, test_ratio=0.15,
                  seed=SEED, test_source=None):
    """Build dataset/<task>/{train,validation,test}.

    Without test_source: source is split into train/validation/test.
    With test_source: it is used unchanged as the test set, and source is split
    into train/validation. Train images identical to a test image are dropped.
    """
    if task not in TASKS:
        raise DatasetError(f"Unknown task '{task}'. Choose from: {list(TASKS)}")
    out = DATASET_DIR / task
    if out.exists() and images_in(out):
        raise DatasetError(f"{out} already contains images. Move or delete them first.")

    pooled = _pool(source, mapping, "train")
    if not pooled:
        raise DatasetError("No images found in the source folder.")
    test_pooled = _pool(test_source, mapping, "test") if test_source else None
    if test_pooled is not None and set(test_pooled) != set(pooled):
        raise DatasetError(f"Classes differ: train {sorted(pooled)} vs test {sorted(test_pooled)}")

    test_hashes = set()
    if test_pooled:
        for items in test_pooled.values():
            test_hashes |= {file_hash(f) for _, f in items}

    rng = random.Random(seed)
    for cls, items in sorted(pooled.items()):
        unique, hashes, leaked = [], set(), 0
        for folder_name, f in items:
            h = file_hash(f)
            if h in hashes:
                continue
            hashes.add(h)
            if h in test_hashes:
                leaked += 1
                continue
            unique.append((folder_name, f))
        rng.shuffle(unique)

        if test_pooled is None:
            n_test = int(len(unique) * test_ratio)
            test_files, rest = unique[:n_test], unique[n_test:]
            n_val = int(len(unique) * val_ratio)
        else:
            seen, test_files = set(), []
            for folder_name, f in test_pooled[cls]:
                h = file_hash(f)
                if h not in seen:
                    seen.add(h)
                    test_files.append((folder_name, f))
            rest = unique
            n_val = int(len(rest) * val_ratio)
        parts = {"test": test_files, "validation": rest[:n_val], "train": rest[n_val:]}
        for split, files in parts.items():
            _copy(files, out / split / cls)
        print(f"{cls:<10} " + "  ".join(f"{s}={len(v)}" for s, v in parts.items())
              + f"   (exact duplicates removed: {len(items) - len(unique) - leaked}; "
              f"train images also found in test removed: {leaked})")
    print(f"\nDone. Now run:  python -m src.prepare_dataset check --task {task}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check", help="validate structure, corrupt files, cross-split duplicates")
    c.add_argument("--task", required=True, choices=list(TASKS))
    s = sub.add_parser("split", help="split a folder of class folders into train/validation/test")
    s.add_argument("--source", required=True)
    s.add_argument("--task", required=True, choices=list(TASKS))
    s.add_argument("--test-source", help="existing test folder; used as-is for the test set")
    s.add_argument("--mapping", help="JSON file: {source_folder_name: target_class_name}")
    s.add_argument("--val", type=float, default=0.15)
    s.add_argument("--test", type=float, default=0.15)
    args = parser.parse_args()
    try:
        if args.command == "check":
            sys.exit(0 if check(args.task) else 1)
        mapping = json.loads(Path(args.mapping).read_text()) if args.mapping else None
        split_dataset(args.source, args.task, mapping, args.val, args.test, test_source=args.test_source)
    except DatasetError as e:
        print(f"\nERROR:\n{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
