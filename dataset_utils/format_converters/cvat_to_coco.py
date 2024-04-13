import datetime as dt
import json
import shutil
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from pathlib import Path

StrPath = str | Path


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--src",
        type=str,
        help="Path to the CVAT dataset directory",
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


def read_cvat_annotation_xml(xml_path: StrPath) -> dict:
    """Read

    Args:
        src_dir (StrPath): _description_
        output_dir (StrPath): _description_

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        dict: _description_
    """

    xml_path = Path(xml_path)

    if not xml_path.exists():
        raise ValueError(f"Annotation XML file does not exist: {xml_path}")

    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    # Read project metadata
    project_meta_el = root.find("./meta/project")
    project_name = project_meta_el.find("name").text
    subsets = project_meta_el.find("subsets").text.split("\n")

    # Read labels of the project
    labels = []
    label_els = project_meta_el.find("labels")
    for label in label_els.findall("label"):
        label_name = label.find("name").text
        label_type = label.find("type").text
        label_color = label.find("color").text

        labels.append(
            {
                "name": label_name,
                "type": label_type,
                "color": label_color,
            }
        )

    # Read tasks of the project
    tasks = []
    task_els = project_meta_el.find("tasks").findall("task")
    for task in task_els:
        task_id = int(task.find("id").text)
        task_name = task.find("name").text
        task_subset = task.find("subset").text
        task_size = int(task.find("size").text)

        tasks.append(
            {
                "id": task_id,
                "name": task_name,
                "subset": task_subset,
                "size": task_size,
            }
        )

    # Read all images element and its attributes
    image_els = root.findall("./image")
    images = []
    annotations = []
    for image_el in image_els:
        img_id = int(image_el.get("id"))
        img_name = image_el.get("name")
        img_subset = image_el.get("subset")
        img_width = int(image_el.get("width"))
        img_height = int(image_el.get("height"))
        img_task_id = int(image_el.get("task_id"))

        images.append(
            {
                "id": img_id,
                "file_name": img_name,
                "width": img_width,
                "height": img_height,
                "subset": img_subset,
                "task_id": img_task_id,
            }
        )

        # Read all annotations of current image
        annotation_els = image_el.findall("./box")
        for annot in annotation_els:
            annot_label = annot.get("label")
            annot_left = float(annot.get("xtl"))
            annot_top = float(annot.get("ytl"))
            annot_right = float(annot.get("xbr"))
            annot_bottom = float(annot.get("ybr"))

            annotations.append(
                {
                    "image_id": img_id,
                    "label": annot_label,
                    "left": annot_left,
                    "top": annot_top,
                    "width": annot_right - annot_left,
                    "height": annot_bottom - annot_top,
                }
            )

    result = {
        "project_name": project_name,
        "subsets": subsets,
        "labels": labels,
        "tasks": tasks,
        "images": images,
        "annotations": annotations,
    }

    return result


def convert_cvat_to_coco(
    src_dir: StrPath,
    output_dir: StrPath,
    force: bool = False,
):
    """Convert dataset from CVAT for images format to COCO format

    References:
        - https://cocodataset.org/#format-data
        - https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/md-coco-overview.html

    Args:
        src_dir (StrPath): directory of the CVAT dataset
        output_dir (StrPath): directory of the output COCO dataset
    """

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

    annot_data = read_cvat_annotation_xml(src_dir / "annotations.xml")

    # Get image path using 'images' in annot_data and check existence
    for img in annot_data["images"]:
        img_path = src_dir / "images" / img["subset"] / img["file_name"]
        if not img_path.exists():
            raise ValueError(f"Image file does not exist: {img_path}")

    # Create output directory
    # Create images and annotations folder
    out_imgs_dir = output_dir / "images"
    out_annots_dir = output_dir / "annotations"
    out_imgs_dir.mkdir(parents=True, exist_ok=False)
    out_annots_dir.mkdir(parents=True, exist_ok=False)

    # Create subset for images
    for subset in annot_data["subsets"]:
        (out_imgs_dir / subset).mkdir(parents=True, exist_ok=True)

    # Get categories for annotations, this was used consistently accross all subsets
    categories = []
    name2id = {}
    for _id, label in enumerate(annot_data["labels"], start=1):
        categories.append(
            {
                "id": _id,
                "name": label["name"],
            }
        )

        name2id[label["name"]] = _id

    # Create annotations for each subset
    for subset in annot_data["subsets"]:
        annot_result = {
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
            "categories": categories,
        }

        # Get images for each subset
        imgs = [img for img in annot_data["images"] if img["subset"] == subset]
        images = []
        for img in imgs:
            images.append(
                {
                    "id": img["id"],  # Image in COCO start from 1
                    "file_name": img["file_name"],
                    "width": img["width"],
                    "height": img["height"],
                }
            )

        subset_img_src_dir = src_dir / "images" / img["subset"]
        subset_img_dst_dir = out_imgs_dir / subset

        shutil.copytree(subset_img_src_dir, subset_img_dst_dir, dirs_exist_ok=True)

        annot_result["images"] = images

        image_ids = set([img["id"] for img in images])

        # Get annotations for each subset
        annotations = [
            annot
            for annot in annot_data["annotations"]
            if annot["image_id"] in image_ids  # Image in COCO start from 1
        ]
        for i in range(len(annotations)):
            annot = annotations[i]

            # Skip if instance not have class name
            if "label" not in annot:
                continue

            annot["id"] = i + 1  # annotation id start from 1

            # label to category_id
            annot["category_id"] = name2id[annot["label"]]
            del annot["label"]

            # left to x
            x = annot["left"]
            del annot["left"]

            # top to y
            y = annot["top"]
            del annot["top"]

            width = annot["width"]
            height = annot["height"]

            annot["bbox"] = [x, y, width, height]
            annot["segmentation"] = []
            annot["iscrowd"] = 0

            annotations[i] = annot

        annot_result["annotations"] = annotations

        # Write to file
        annot_file_name = "instances_{}.json".format(subset)
        with open(out_annots_dir / annot_file_name, "w") as f:
            json.dump(annot_result, f, indent=2)


def main():
    args = get_args()

    convert_cvat_to_coco(
        src_dir=args.src,
        output_dir=args.output,
        force=args.force,
    )


if __name__ == "__main__":
    main()
