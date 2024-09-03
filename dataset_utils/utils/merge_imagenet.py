import os
import sys

import yaml
from loguru import logger

# Get the current directory
current_dir = os.getcwd()  # Use os.getcwd() instead of %cd%

# Add the current directory to the search path
sys.path.append(current_dir)


import shutil
from argparse import ArgumentParser
from pathlib import Path

from imagenet_util import read_imagenet

StrPath = str | Path


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--src",
        action="append",
        type=str,
        help="Path to the ImageNet dataset directory",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to the output directory",
        required=True,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output directory",
    )

    return parser.parse_args()


def merge_imagenet_datasets(
    src_dirs: list[StrPath],
    output_dir: StrPath,
    force: bool = False,
):
    """Merge ImageNet datasets into one new dataset

    Args:
        src_dirs (list[StrPath]): list of directory of the ImageNet dataset
        output_dir (StrPath): directory of the output dataset
        force (bool, optional): Overwrite existing output directory. Defaults to False.
    """  # noqa: E501

    src_dirs = [Path(src_dir) for src_dir in src_dirs]
    for src_dir in src_dirs:
        src_dir = Path(src_dir)
        if not src_dir.exists():
            raise ValueError(f"Source directory does not exist: {src_dir}")
        if not src_dir.is_dir():
            raise ValueError(f"Source is not a directory: {src_dir}")

    output_dir = Path(output_dir)
    if not force:
        if output_dir.exists():
            raise ValueError("Output directory already exists")
    else:
        print("Output directory already exists. Removing existing output directory")
        shutil.rmtree(str(output_dir), ignore_errors=True)
        print("Creating new output directory")

    print("Scanning source directorie label...")
    subsets: list[str] | None = None
    names: list[str] | None = None
    name2id: dict[str, int] = {}
    dataset_infos: list[tuple[Path, dict]] = []
    for src_dir in src_dirs:
        src_dir = Path(src_dir)

        dataset_data = read_imagenet(src_dir)
        dataset_infos.append((src_dir, dataset_data))
        if names is None:
            names = dataset_data["names"]
            name2id = {name: i for i, name in enumerate(names)} # type: ignore
        else:
            if (
                set(names).difference(set(dataset_data["names"]))
                or set(dataset_data["names"]).difference(set(names))
            ):
                logger.error("Names in the datasets do not match: %s - " % (set(names), set(dataset_data["names"])))
                raise ValueError(
                    f"Names in the datasets do not match: {set(names)} - {set(dataset_data['names'])}"
                )

        if subsets is None:
            subsets = dataset_data["subsets"]
        else:
            if (
                set(subsets).difference(dataset_data["subsets"])
                or dataset_data["subsets"].difference(set(subsets))
            ):
                print(set(subsets).difference(dataset_data["subsets"]))
                logger.error("Subsets in the datasets do not match: %s - %s" % (set(subsets), dataset_data['subsets']))
                raise ValueError(
                    f"Subsets in the datasets do not match: {set(subsets)} - {set(dataset_data['subsets'])}"
                )

    # start to merge dataset
    for src_dir, dataset_info in dataset_infos:
        for subset in dataset_info["subsets"]:
            subset_dir = output_dir / subset
            subset_dir.mkdir(parents=True, exist_ok=True)
            for img_info in dataset_info[subset]:
                src_img = src_dir / subset / img_info["label"] / img_info["filename"]
                dst_img = output_dir / subset / img_info["label"] / img_info["filename"]
                dst_img.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src_img, dst_img)

    if subsets is None:
        raise ValueError("No subsets found")

    # writing metadata to yaml file
    data_yml = {}
    for subset in subsets:
        data_yml[subset] = f"./images/{subset}"

    data_yml.update(
        {
            "nc": len(name2id),
            "names": {v: k for k, v in name2id.items()},
        }
    )

    with open(output_dir / "data.yaml", "w") as f:
        yaml.dump(data_yml, f, sort_keys=False)


def main():
    args = get_args()

    # we need at least two source directories
    if len(args.src) < 2:
        raise ValueError("At least two source directories are required")

    merge_imagenet_datasets(
        src_dirs=args.src,
        output_dir=args.output,
        force=args.force,
    )


if __name__ == "__main__":
    main()
