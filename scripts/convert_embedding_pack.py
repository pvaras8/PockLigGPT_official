import argparse
import shutil
import zipfile
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract a memory-mappable NPY array from a legacy NPZ pack."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete the NPZ after the NPY has been extracted and validated.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")

    with zipfile.ZipFile(args.input) as archive:
        member_name = "emb_stack.npy"
        if member_name not in archive.namelist():
            raise ValueError(f"{args.input} does not contain {member_name}")

        with archive.open(member_name) as source:
            with args.output.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)

    emb_stack = np.load(args.output, mmap_mode="r")
    if emb_stack.ndim != 2:
        args.output.unlink(missing_ok=True)
        raise ValueError(f"Expected a 2D embedding stack, got {emb_stack.shape}")

    print(
        f"Extracted {args.output}: shape={emb_stack.shape}, dtype={emb_stack.dtype}"
    )

    if args.delete_source:
        args.input.unlink()
        print(f"Deleted source archive: {args.input}")


if __name__ == "__main__":
    main()
