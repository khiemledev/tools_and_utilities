import json
import os
import sys

# Get the current directory
current_dir = os.getcwd()  # Use os.getcwd() instead of %cd%

# Add the current directory to the search path
sys.path.append(current_dir)


import datetime as dt
import shutil
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from pathlib import Path

import imagesize
from utils.bbox_utils import yolo2xywh
from yolo_utils import read_yolo_data_yaml, validate_dataset_folder

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
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip missing images/labels",
    )

    return parser.parse_args()


def convert_yolo_ultralytics_to_coco(
    src_dir: StrPath,
    output_dir: StrPath,
    force: bool = False,
    skip_missing: bool = False,
):
    """Convert dataset from YOLO Ultralytics format to COCO format

    References:
        - https://docs.ultralytics.com/datasets/detect/

    Args:
        src_dir (StrPath): directory of the CVAT dataset
        output_dir (StrPath): directory of the output YOLO Ultralytics dataset
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

    data_yml = read_yolo_data_yaml(data_yml_file)
    ds_data = validate_dataset_folder(data_yml, src_dir, skip_missing)

    print(data_yml)

    # Create output directory
    annotations_output_dir = output_dir / "annotations"
    images_output_dir = output_dir / "images"
    annotations_output_dir.mkdir(parents=True, exist_ok=True)
    images_output_dir.mkdir(parents=True, exist_ok=True)

    # Convert names to categories for COCO
    class_names = {}
    for idx, name in enumerate(data_yml["names"], start=1):
        class_names[idx] = name

    for subset, data in ds_data.items():
        subset_info = {
            "info": {
                "description": "COCO Dataset",
                "url": "https://khiemle.dev",
                "version": "1.0",
                "year": dt.datetime.now().year,
                "contributor": "Khiem Le",
                "date_created": dt.datetime.now().strftime("%Y/%m/%d"),
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [{"id": k, "name": v} for k, v in class_names.items()],
        }

        for img in data:
            img_path = Path(img["image"])
            labels = img["labels"]

            # Copy image to output directory
            shutil.copy(img_path, images_output_dir / img_path.name)

            # Get image size
            imw, imh = imagesize.get(str(img_path))

            # Add image info
            subset_info["images"].append({
                "id": len(subset_info["images"]) + 1,
                "file_name": img_path.name,
                "width": imw,
                "height": imh,
            })

            # Convert labels to COCO format
            for label in labels:
                x, y, w, h = yolo2xywh(*label[1:], imw, imh)
                category_id = label[0] + 1
                annotation = {
                    "id": len(subset_info["annotations"]) + 1,
                    "image_id": len(subset_info["images"]),
                    "category_id": category_id,
                    "segmentation": [],
                    "area": w * h,
                    "bbox": [x, y, w, h],
                    "iscrowd": 0,
                }
                subset_info["annotations"].append(annotation)

        # Write result to json output
        json_output_path = annotations_output_dir / f"instances_{subset}.json"
        with json_output_path.open("w") as f:
            json.dump(subset_info, f, indent=2)


def main():
    args = get_args()

    convert_yolo_ultralytics_to_coco(
        src_dir=args.src,
        output_dir=args.output,
        force=args.force,
        skip_missing=args.skip_missing,
    )


if __name__ == "__main__":
    main()
