import io
import os
from argparse import ArgumentParser
from pathlib import Path
from zipfile import ZipFile


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--root-dir",
        type=str,
        help="Path to the root directory",
        required=True,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to the output zip file",
        required=True,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output file",
    )
    return parser.parse_args()


def zip_files(
    root_dir: Path,
    output_path: Path,
    force: bool = False,
):
    if not root_dir.exists():
        raise ValueError(f"Root directory does not exist: {root_dir}")

    if not root_dir.is_dir():
        raise ValueError(f"Root directory is not a directory: {root_dir}")

    if output_path.suffix != ".zip":
        raise ValueError(f"Output path is not a zip file: {output_path}")

    if not force and output_path.exists():
        raise ValueError(f"Output file already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Zip files which relative to root_dir
    with ZipFile(output_path, "w") as zip_file:
        for root, dirs, files in os.walk(str(root_dir)):
            for file in files:
                path = Path(root, file)
                zip_file.write(path, Path(path).relative_to(root_dir))


def main():
    args = get_args()

    zip_files(
        Path(args.root_dir),
        Path(args.output_path),
        args.force,
    )

if __name__ == "__main__":
    main()
