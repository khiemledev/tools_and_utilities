import datetime as dt
import json
import shutil
from argparse import ArgumentParser
from pathlib import Path

from cvat_utils import read_cvat_annotation_xml

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

    # arguments to map subsets. Example --subset-map Train:train --subset-map Test:test
    parser.add_argument(
        "--subset-map",
        type=str,
        action="append",
        help="Map subset names to new subset names",
        default=[],
    )

    return parser.parse_args()


def convert_cvat_to_coco(
    src_dir: StrPath,
    output_dir: StrPath,
    force: bool = False,
    subset_map: dict[str, str] | None = None,
):
    """Convert dataset from CVAT for images format to COCO format

    References:
        - https://cocodataset.org/#format-data
        - https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/md-coco-overview.html

    Args:
        src_dir (StrPath): directory of the CVAT dataset
        output_dir (StrPath): directory of the output COCO dataset.
        force (bool, optional): Overwrite existing output directory. Defaults to False.
        subset_map (dict[str, str], optional): Map subset names to new subset names. Defaults to None.
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

    # Validate subset map if provided
    if subset_map:
        for src, target in subset_map.items():
            if src not in annot_data["subsets"]:
                raise ValueError(f"Subset '{src}' does not exist in CVAT dataset")
            if target in annot_data["subsets"]:
                raise ValueError(f"Subset '{target}' already exists in CVAT dataset")

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

        # map to new subset if provided
        new_subset = subset
        if subset_map and subset in subset_map:
            new_subset = subset_map[subset]

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
        subset_img_dst_dir = out_imgs_dir / new_subset

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
        annot_file_name = "instances_{}.json".format(new_subset)
        with open(out_annots_dir / annot_file_name, "w") as f:
            json.dump(annot_result, f, indent=2)


def main():
    args = get_args()

    subset_map = {}
    for _map in args.subset_map:
        k, v = _map.split(":")
        subset_map[k] = v

    print(subset_map)

    convert_cvat_to_coco(
        src_dir=args.src,
        output_dir=args.output,
        force=args.force,
        subset_map=subset_map,
    )


if __name__ == "__main__":
    main()
