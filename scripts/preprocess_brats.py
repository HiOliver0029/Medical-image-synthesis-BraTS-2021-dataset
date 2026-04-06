from __future__ import annotations

import argparse
import os
import tarfile
from pathlib import Path

from dlmi_hw1.data.brats_io import collect_patient_volumes
from dlmi_hw1.data.preprocess import PreprocessConfig, preprocess_brats_2d


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    project_candidate = PROJECT_ROOT / path
    cwd_candidate = Path.cwd() / path

    if project_candidate.exists():
        return project_candidate
    if cwd_candidate.exists():
        return cwd_candidate
    return project_candidate


def maybe_extract_tars(raw_root: Path) -> Path:
    nii_files = []
    tar_files = []
    for root, _, files in os.walk(raw_root, followlinks=True):
        for file_name in files:
            file_path = Path(root) / file_name
            if file_name.endswith(".nii.gz"):
                nii_files.append(file_path)
            elif file_name.endswith(".tar"):
                tar_files.append(file_path)

    if nii_files:
        return raw_root

    tar_files.sort()
    if not tar_files:
        return raw_root

    extracted_root = raw_root / "extracted"
    extracted_root.mkdir(parents=True, exist_ok=True)

    print(f"No .nii.gz found in {raw_root}. Extracting tar archives to {extracted_root}...")
    for tar_path in tar_files:
        marker = extracted_root / f".{tar_path.name}.done"
        if marker.exists():
            continue
        print(f"Extracting: {tar_path.name}")
        with tarfile.open(tar_path, "r") as tf:
            tf.extractall(path=extracted_root)
        marker.touch()

    return extracted_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess BraTS into paired 2D slices")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--source-modality", type=str, default="t1")
    parser.add_argument("--target-modality", type=str, default="t2")
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--min-foreground-ratio", type=float, default=0.1)
    parser.add_argument("--axis", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="Only report detected patient count and exit",
    )
    args = parser.parse_args()

    raw_root = resolve_path(args.raw_root)
    processed_root = resolve_path(args.processed_root)
    raw_root = maybe_extract_tars(raw_root)

    if args.sanity_check:
        patients = collect_patient_volumes(raw_root)
        valid = [
            p
            for p in patients
            if args.source_modality.lower() in p.paths and args.target_modality.lower() in p.paths
        ]
        print("Sanity check")
        print("Raw root:", raw_root)
        print("Detected patients (all):", len(patients))
        print(
            f"Detected patients with {args.source_modality.lower()} and {args.target_modality.lower()}:",
            len(valid),
        )
        return

    cfg = PreprocessConfig(
        raw_root=raw_root,
        processed_root=processed_root,
        source_modality=args.source_modality.lower(),
        target_modality=args.target_modality.lower(),
        split_ratio=args.split_ratio,
        min_foreground_ratio=args.min_foreground_ratio,
        axis=args.axis,
        seed=args.seed,
    )

    counts = preprocess_brats_2d(cfg)
    print("Preprocessing complete")
    print("Train slices:", counts["train"])
    print("Val slices:", counts["val"])


if __name__ == "__main__":
    main()
