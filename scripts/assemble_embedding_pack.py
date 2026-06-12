import argparse
import shutil
import zipfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reassemble a split ProtT5 embedding NPZ archive."
    )
    parser.add_argument(
        "--parts-dir",
        required=True,
        type=Path,
        help="Directory containing per_residue_pack.npz.part-* files.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--delete-parts",
        action="store_true",
        help="Delete the downloaded parts after validating the NPZ.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    parts = sorted(args.parts_dir.glob("per_residue_pack.npz.part-*"))
    if not parts:
        raise FileNotFoundError(
            f"No per_residue_pack.npz.part-* files found in {args.parts_dir}"
        )
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as destination:
        for part in parts:
            print(f"Appending {part.name}")
            with part.open("rb") as source:
                shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)

    with zipfile.ZipFile(args.output) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            args.output.unlink(missing_ok=True)
            raise ValueError(f"Corrupt member after assembly: {bad_member}")
        if "emb_stack.npy" not in archive.namelist():
            args.output.unlink(missing_ok=True)
            raise ValueError("Assembled NPZ does not contain emb_stack.npy")

    print(f"Assembled and validated: {args.output}")

    if args.delete_parts:
        for part in parts:
            part.unlink()
        print(f"Deleted {len(parts)} source parts")


if __name__ == "__main__":
    main()
