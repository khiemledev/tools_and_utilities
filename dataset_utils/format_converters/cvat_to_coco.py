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

    # arguments to resplit dataset subsets. Example --split-ratio train:0.8 --split-ratio val:0.2
    parser.add_argument(
        "--split-ratio",
        type=str,
        action="append",
        help="Split dataset into subsets and specify the ratio of each subset",
        default=[],
    )

    return parser.parse_args()


def convert_cvat_to_coco(
    src_dir: StrPath,
    output_dir: StrPath,
    force: bool = False,
    subset_map: dict[str, str] | None = None,
    split_ratio: dict[str, float] | None = None,
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
        split_ratio (dict[str, float], optional): Split dataset into subsets and specify the ratio of each subset. Defaults to None.
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
    name2id = {}
    for _id, label in enumerate(annot_data["labels"], start=1):
        name2id[label["name"]] = _id

    # Create annotations for each subset
    # We will get the images along with its annotations before actually writing
    # the annotations to the file. This will support resplitting
    all_images: dict[str, dict] = {}

    # Create annotations for each subset
    for subset in annot_data["subsets"]:
        # Get images for each subset
        imgs = [img for img in annot_data["images"] if img["subset"] == subset]
        images: dict[int, dict] = {}  # id: data
        for img in imgs:
            images[img["id"]] = {
                "id": img["id"],
                "file_name": img["file_name"],
                "width": img["width"],
                "height": img["height"],
                "subset": img["subset"],
            }

            all_images[f"{img['id']}_{img['subset']}"] = {
                "image": images[img["id"]],
                "annotations": [],
                "subset": img["subset"],
            }

        image_ids = set(images.keys())

        # Process annotations of each subset. Each each annotation contains
        # image_id, cls_id and bbox
        annots = [
            annot
            for annot in annot_data["annotations"]
            if annot["image_id"] in image_ids and annot["subset"] == subset  # Image in COCO start from 1
        ]

        annotations: dict[int, dict] = {}  # image_id: data
        for i in range(len(annots)):
            annot = annots[i]

            # # Skip if instance not have class name
            if "label" not in annot:
                continue

            annot["id"] = i

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

            annots[i] = annot

            annotations[annot["image_id"]] = annot

            all_images[f"{annot["image_id"]}_{subset}"]["annotations"].append(
                annotations[annot["image_id"]],
            )

    for k, data in all_images.items():
        if len(data.get("annotations", [])) == 0:
            raise ValueError(f"No annotations found for image: {k}")

    # if split_ratio is provided, calculate size for each subset first
    if split_ratio:
        keys = list(all_images.keys())
        prev = 0
        for subset, ratio in split_ratio.items():
            split_size = round(len(all_images) * ratio)

            for key in keys[prev : prev + split_size]:
                all_images[key]["subset"] = subset

            prev += split_size

    # copy images to output directory
    for subset in annot_data["subsets"]:
        subset_img_src_dir = src_dir / "images" / subset
        subset_img_dst_dir = out_imgs_dir

        shutil.copytree(subset_img_src_dir, subset_img_dst_dir, dirs_exist_ok=True)

    # Get the new subsets if split_ratio or subset_map is provided
    subsets = set(annot_data["subsets"])
    if split_ratio:
        subsets = set(split_ratio.keys())
    elif subset_map:
        for src, target in subset_map.items():
            if src in subsets:
                subsets.remove(src)
                subsets.add(target)

    subsets_data: dict[str, dict] = {} # subset: data

    # adding images to subsets_data
    for key, data in all_images.items():
        img_id, subset = key.split("_")

        # map to new subset if provided
        new_subset = subset

        if split_ratio:
            new_subset = data["subset"]
        elif subset_map and subset in subset_map:
            new_subset = subset_map[subset]

        if new_subset not in subsets_data:
            subsets_data[new_subset] = {
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
                "categories": [{"id": v, "name": k} for k, v in name2id.items()],
            }

        # If resplit the data, reset the id of the image
        if split_ratio:
            data["image"]["id"] = len(subsets_data[new_subset]["images"]) + 1

        subsets_data[new_subset]["images"].append(data["image"])

    # write annotations to file
    for key, data in all_images.items():
        img_id, subset = key.split("_")

        img = data["image"]
        annotations = data["annotations"]

        # map to new subset if provided
        new_subset = subset

        if split_ratio:
            new_subset = data["subset"]
        elif subset_map and subset in subset_map:
            new_subset = subset_map[subset]

        print(new_subset)

        # Set image id for annotations
        for i in range(len(data["annotations"])):
            annot = data["annotations"][i]
            annot["image_id"] = img["id"]

            # reset the id of the annotation
            annot["id"] = len(subsets_data[new_subset]["annotations"]) + 1

            subsets_data[new_subset]["annotations"].append(annot)


    for subset in subsets_data:
        with open(out_annots_dir / f"instances_{subset}.json", "w") as f:
            json.dump(subsets_data[subset], f, indent=2)



def main():
    args = get_args()

    subset_map = {}
    for _map in args.subset_map:
        k, v = _map.split(":")
        subset_map[k] = v

    # Process split ratio
    split_ratio = {}
    for arg in args.split_ratio:
        k, v = arg.split(":")
        split_ratio[k] = float(v)

    FAULT_TOLERANCE = 1e-6
    if split_ratio and (1 - sum(split_ratio.values())) > FAULT_TOLERANCE:
        raise ValueError("Sum of split ratios should be 1.0")

    if split_ratio and subset_map:
        raise ValueError("Subset map and split ratio cannot be used together")

    convert_cvat_to_coco(
        src_dir=args.src,
        output_dir=args.output,
        force=args.force,
        subset_map=subset_map,
        split_ratio=split_ratio,
    )


if __name__ == "__main__":
    main()
