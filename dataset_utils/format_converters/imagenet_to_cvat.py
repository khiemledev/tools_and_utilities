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
from imagenet_utils import read_imagenet
from utils.bbox_utils import yolo2xyxy
from yolo_utils import read_yolo_data_yaml, validate_dataset_folder

StrPath = str | Path


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--src",
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


def convert_imagenet_to_cvat(
    src_dir: StrPath,
    output_dir: StrPath,
    force: bool = False,
):
    """Convert dataset from ImageNet format to CVAT for images format

    Args:
        src_dir (StrPath): directory of the ImageNet dataset
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

    imnet_data = read_imagenet(src_dir)

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
    for name in imnet_data["names"]:
        label_el = ET.SubElement(labels_el, "label")
        name_el = ET.SubElement(label_el, "name")
        name_el.text = name
        type_el = ET.SubElement(label_el, "type")
        type_el.text = "tag"

        # Empty elements
        # ET.SubElement(label_el, "color")
        ET.SubElement(label_el, "attributes")

    subsets_el = ET.SubElement(meta_project_el, "subsets")
    subsets_el.text = "\n".join(imnet_data["subsets"])

    dumped_meta_el = ET.SubElement(meta_el, "dumped")
    dumped_meta_el.text = dt.datetime.today().strftime("%Y-%m-%d %H:%M:%S.%f%z")

    # Start to process data and prepare to write to yaml file
    for subset in imnet_data["subsets"]:
        data: list[dict] = imnet_data[subset]

        # Create subset image output directory
        subset_subset_imgs_out_dir = output_dir / "images" / subset
        subset_subset_imgs_out_dir.mkdir(parents=True, exist_ok=True)

        for img_data in data:
            image_path = img_data["file_path"]
            label = img_data["label"]

            # Copy image to output subset dir
            output_img_path = subset_subset_imgs_out_dir / Path(image_path).name
            # print(image_path, Path(image_path).exists(), output_img_path)
            shutil.copyfile(image_path, output_img_path)

            # Set image element's attributes
            # imw, imh = imagesize.get(str(image_path))
            imw, imh = img_data["width"], img_data["height"]
            image_el = ET.SubElement(root, "image")
            image_el.set("id", str(img_data["id"]))
            image_el.set("name", Path(image_path).name)
            image_el.set("subset", subset)
            image_el.set("width", str(imw))
            image_el.set("height", str(imh))

            tag_el = ET.SubElement(image_el, "tag")
            tag_el.set("label", label)
            tag_el.set("source", "manual")

    # This is not beaultifuly indented
    tree = ET.ElementTree(root)
    ET.indent(tree)
    tree.write(str(annotation_xml_path), encoding="utf-8", xml_declaration=True)


def main():
    args = get_args()

    convert_imagenet_to_cvat(
        src_dir=args.src,
        output_dir=args.output,
        force=args.force,
    )


if __name__ == "__main__":
    main()
