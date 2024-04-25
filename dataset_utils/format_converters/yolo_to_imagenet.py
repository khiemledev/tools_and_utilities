import os
import sys

# Get the current directory
current_dir = os.getcwd()  # Use os.getcwd() instead of %cd%

# Add the current directory to the search path
sys.path.append(current_dir)


import shutil
from argparse import ArgumentParser
from pathlib import Path

from PIL import Image
from utils.bbox_utils import yolo2xyxy
from yolo_utils import read_data_yaml, validate_dataset_folder

StrPath = str | Path


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--src",
        type=str,
        help="Path to the YOLO dataset directory",
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


def convert_yolo_ultralytics_to_imagenet(
    src_dir: StrPath,
    output_dir: StrPath,
    force: bool = False,
):
    """Convert dataset from YOLO Ultralytics format to ImageNet format

    References:
        - https://docs.ultralytics.com/datasets/detect/

    Args:
        src_dir (StrPath): directory of the YOLO Ultralytics dataset
        output_dir (StrPath): directory of the output dataset
        force (bool, optional): Overwrite existing output directory. Defaults to False.
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

    data_yml_file = src_dir / "data.yaml"
    if not data_yml_file.exists():
        raise ValueError(f"data.yaml does not exist: {data_yml_file}")

    data_yml = read_data_yaml(data_yml_file)
    ds_data = validate_dataset_folder(data_yml, src_dir)

    idx2name = {k: v for k, v in enumerate(data_yml.get("names"))}

    for subset, data in ds_data.items():
        subset_dir = output_dir / subset

        # make dir for each class name
        for class_name in data_yml.get("names"):
            (subset_dir / class_name).mkdir(parents=True, exist_ok=True)

        # crop images
        for img_data in data:
            img_path = Path(img_data["image"])
            labels = img_data["labels"]

            img = Image.open(img_path)
            imw, imh = img.size
            for i, (cls_id, xc, yc, w, h) in enumerate(labels):
                x1, y1, x2, y2 = yolo2xyxy(xc, yc, w, h, imw, imh)
                roi = img.crop((x1, y1, x2, y2))

                roi_save_path = (
                    subset_dir / idx2name[cls_id] / f"{img_path.stem}_{i}.jpg"
                )
                roi.save(roi_save_path)
                print(roi_save_path)


def main():
    args = get_args()

    convert_yolo_ultralytics_to_imagenet(
        src_dir=args.src,
        output_dir=args.output,
        force=args.force,
    )


if __name__ == "__main__":
    main()
