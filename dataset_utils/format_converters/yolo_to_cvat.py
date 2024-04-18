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


def convert_yolo_ultralytics_to_cvat(
    src_dir: StrPath,
    output_dir: StrPath,
    force: bool = False,
):
    """Convert dataset from YOLO Ultralytics format to CVAT for images format

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

    data_yml = read_data_yaml(data_yml_file)
    ds_data = validate_dataset_folder(data_yml, src_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=not force)

    # Write result to xml file
    annotation_xml_path = output_dir / "annotations.xml"
    root = ET.Element("annotations")

    # CVAT for images version 1.1
    version_el = ET.SubElement(root, "version")
    version_el.text = "1.1"

    meta_el = ET.SubElement(root, "meta")

    meta_project_el = ET.SubElement(meta_el, "project")

    # add labels to project
    labels_el = ET.SubElement(meta_project_el, "labels")
    for name in data_yml["names"]:
        label_el = ET.SubElement(labels_el, "label")
        name_el = ET.SubElement(label_el, "name")
        name_el.text = name
        type_el = ET.SubElement(label_el, "type")
        type_el.text = "rectangle"

        # Empty elements
        # ET.SubElement(label_el, "color")
        ET.SubElement(label_el, "attributes")

    dumped_meta_el = ET.SubElement(meta_el, "dumped")
    dumped_meta_el.text = dt.datetime.today().strftime("%Y-%m-%d %H:%M:%S.%f%z")

    # Start to process data and prepare to write to yaml file
    for subset, data in ds_data.items():
        # Create subset image output directory
        subset_subset_imgs_out_dir = output_dir / "images" / subset
        subset_subset_imgs_out_dir.mkdir(parents=True)

        for idx, img_data in enumerate(data):
            image_path = img_data["image"]
            labels = img_data["labels"]

            # Copy image to output subset dir
            output_img_path = subset_subset_imgs_out_dir / Path(image_path).name
            shutil.copyfile(image_path, output_img_path)

            # Set image element's attributes
            imw, imh = imagesize.get(str(image_path))
            image_el = ET.SubElement(root, "image")
            image_el.set("id", str(idx))
            image_el.set("name", Path(image_path).name)
            image_el.set("subset", subset)
            image_el.set("width", str(imw))
            image_el.set("height", str(imh))
            image_el.set("z_order", "0")

            # Append boxes to image
            for label in labels:
                # label: cls_id, xcn, ycn, bwn, bhn
                x1, y1, x2, y2 = yolo2xyxy(
                    *label[1:],
                    imw,
                    imh,
                )

                box_el = ET.SubElement(image_el, "box")
                box_el.set("occluded", "0")
                box_el.set("label", data_yml["names"][label[0]])
                box_el.set("xtl", str(x1))
                box_el.set("ytl", str(y1))
                box_el.set("xbr", str(x2))
                box_el.set("ybr", str(y2))

    # This is not beaultifuly indented
    tree = ET.ElementTree(root)
    ET.indent(tree)
    tree.write(str(annotation_xml_path), encoding="utf-8", xml_declaration=True)


def main():
    args = get_args()

    convert_yolo_ultralytics_to_cvat(
        src_dir=args.src,
        output_dir=args.output,
        force=args.force,
    )


if __name__ == "__main__":
    main()
