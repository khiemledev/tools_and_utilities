import os
import sys

from matplotlib import path

# Get the current directory
current_dir = os.getcwd()  # Use os.getcwd() instead of %cd%

# Add the current directory to the search path
sys.path.append(current_dir)


import random
import shutil
from argparse import ArgumentParser
from pathlib import Path

import yaml
from cvat_utils import read_cvat_annotation_xml
from utils.bbox_utils import xywh2yolo

StrPath = str | Path


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--src",
        type=str,
        help="Path to the images directory",
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

    # arguments to resplit dataset subsets. Example --split-ratio train:0.8 --split-ratio val:0.2
    parser.add_argument(
        "--split-ratio",
        type=str,
        action="append",
        help="Split dataset into subsets and specify the ratio of each subset",
        default=[],
    )

    return parser.parse_args()


def convert_images_dir_to_yolo_ultralytics(
    src_dir: StrPath,
    output_dir: StrPath,
    force: bool = False,
    split_ratio: dict[str, float] | None = None,
):
    """Convert images dir to YOLO Ultralytics format

    References:
        - https://docs.ultralytics.com/datasets/detect/

    Args:
        src_dir (StrPath): directory of the CVAT dataset
        output_dir (StrPath): directory of the output YOLO Ultralytics dataset
        force (bool, optional): Overwrite existing output directory. Defaults to False.
        split_ratio (dict[str, float], optional): Split dataset into subsets and specify the ratio of each subset. Defaults to None.
    """  # noqa: E501

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

    # List all images
    src_imgs = [f for f in src_dir.iterdir() if f.is_file()]
    all_images = {}
    for idx, path in enumerate(src_imgs):
        all_images[str(idx)] = {
            "file_name": path.name,
            "img_path": path,
        }

    # Create output directory
    # Create images and annotations folder
    out_imgs_dir = output_dir / "images"
    out_annots_dir = output_dir / "labels"
    out_imgs_dir.mkdir(parents=True, exist_ok=False)
    out_annots_dir.mkdir(parents=True, exist_ok=False)

    if split_ratio:
        keys = list(all_images.keys())

        # make a list of position to split
        split_pos = []

        for subset, ratio in split_ratio.items():
            split_size = round(len(all_images) * ratio)
            split_pos.extend([subset] * split_size)

        # shuffle the list
        random.shuffle(split_pos)
        for key in keys:
            all_images[key]["subset"] = split_pos.pop()

    subsets = {"train": 1.0}
    if split_ratio:
        subsets = set(split_ratio.keys())

    # Copy images and write annotations to file
    for img_id, data in all_images.items():
        new_subset = data.get("subset", "train")

        # Make subdir for each subset
        subset_img_dst_dir = out_imgs_dir / new_subset
        subset_annot_dst_dir = out_annots_dir / new_subset

        subset_img_dst_dir.mkdir(parents=True, exist_ok=True)
        subset_annot_dst_dir.mkdir(parents=True, exist_ok=True)

        # Write annotations for each image
        # Copy image
        src_img_path = Path(data["img_path"])
        shutil.copy(str(src_img_path), str(subset_img_dst_dir / data["file_name"]))

        # Write annotations to txt file
        output_txt_file = (subset_annot_dst_dir / data["file_name"]).with_suffix(".txt")
        output_txt_file.touch()

    # Create data.yaml file
    data_yml = {}
    for subset in subsets:
        data_yml[subset] = f"./images/{subset}"

    name2id = {
        "class1": 0,
        "class2": 1,
    }
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

    # Process split ratio
    split_ratio = {}
    for arg in args.split_ratio:
        k, v = arg.split(":")
        split_ratio[k] = float(v)

    FAULT_TOLERANCE = 1e-6
    if split_ratio and (1 - sum(split_ratio.values())) > FAULT_TOLERANCE:
        raise ValueError("Sum of split ratios should be 1.0")

    convert_images_dir_to_yolo_ultralytics(
        src_dir=args.src,
        output_dir=args.output,
        force=args.force,
        split_ratio=split_ratio,
    )


if __name__ == "__main__":
    main()
