from __future__ import annotations

import argparse
from pathlib import Path

import kagglehub


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download BraTS 2021 Task1 dataset via KaggleHub")
    parser.add_argument(
        "--dataset",
        type=str,
        default="dschettler8845/brats-2021-task1",
        help="Kaggle dataset slug",
    )
    parser.add_argument(
        "--copy-to",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Optional copy destination inside project",
    )
    args = parser.parse_args()

    path = kagglehub.dataset_download(args.dataset)
    print("Path to dataset files:", path)

    src = Path(path)
    dst = args.copy_to if args.copy_to.is_absolute() else PROJECT_ROOT / args.copy_to
    dst.mkdir(parents=True, exist_ok=True)

    if src.resolve() != dst.resolve():
        # Symlink keeps storage low and still gives a stable project-local path.
        link = dst / "brats_2021_task1"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(src, target_is_directory=True)
        print("Created symlink:", link, "->", src)


if __name__ == "__main__":
    main()
