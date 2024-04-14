import shutil
from argparse import ArgumentParser
from pathlib import Path

import yaml
from cvat_utils import read_cvat_annotation_xml


def xywh2yolo(
    x1: int,
    y1: int,
    box_w: int,
    box_h: int,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    x_center = (x1 + box_w / 2) / img_w
    y_center = (y1 + box_h / 2) / img_h
    w = box_w / img_w
    h = box_h / img_h
    return x_center, y_center, w, h


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


def convert_cvat_to_yolo_ultralytics(
    src_dir: StrPath,
    output_dir: StrPath,
    force: bool = False,
    subset_map: dict[str, str] | None = None,
):
    """Convert dataset from CVAT for images format to YOLO Ultralytics format

    References:
        - https://docs.ultralytics.com/datasets/detect/

    Args:
        src_dir (StrPath): directory of the CVAT dataset
        output_dir (StrPath): directory of the output YOLO Ultralytics dataset
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

    # Get image path using 'images' in annot_data and check existence
    for img in annot_data["images"]:
        img_path = src_dir / "images" / img["subset"] / img["file_name"]
        if not img_path.exists():
            raise ValueError(f"Image file does not exist: {img_path}")

    # Validate subset map if provided
    if subset_map:
        for src, target in subset_map.items():
            if src not in annot_data["subsets"]:
                raise ValueError(f"Subset '{src}' does not exist in CVAT dataset")
            if target in annot_data["subsets"]:
                raise ValueError(f"Subset '{target}' already exists in CVAT dataset")

    # Create output directory
    # Create images and annotations folder
    out_imgs_dir = output_dir / "images"
    out_annots_dir = output_dir / "labels"
    out_imgs_dir.mkdir(parents=True, exist_ok=False)
    out_annots_dir.mkdir(parents=True, exist_ok=False)

    # Get categories for annotations, this was used consistently accross all subsets
    name2id = {}
    for _id, label in enumerate(annot_data["labels"]):
        name2id[label["name"]] = _id

    # Create annotations for each subset
    for subset in annot_data["subsets"]:
        # map to new subset if provided
        new_subset = subset
        if subset_map and subset in subset_map:
            new_subset = subset_map[subset]

        # Get images for each subset
        imgs = [img for img in annot_data["images"] if img["subset"] == subset]
        images: dict[int, dict] = {}  # id: data
        for img in imgs:
            images[img["id"]] = {
                "file_name": img["file_name"],
                "width": img["width"],
                "height": img["height"],
            }

        # Make subdir for each subset
        subset_img_src_dir = src_dir / "images" / img["subset"]
        subset_img_dst_dir = out_imgs_dir / new_subset
        subset_annot_dst_dir = out_annots_dir / new_subset
        print(subset, new_subset)
        print(subset_img_dst_dir, subset_annot_dst_dir)

        subset_img_dst_dir.mkdir(parents=True, exist_ok=True)
        subset_annot_dst_dir.mkdir(parents=True, exist_ok=True)

        # Process annotations of each subset
        image_ids = set(images.keys())
        annots = [
            annot
            for annot in annot_data["annotations"]
            if annot["image_id"] in image_ids  # Image in COCO start from 1
        ]
        annotations: dict[int, dict] = {}  # image_id: data
        for annot in annots:
            x = annot["left"]
            y = annot["top"]
            width = annot["width"]
            height = annot["height"]

            img_data = images[annot["image_id"]]
            imw = img_data["width"]
            imh = img_data["height"]

            annotations[annot["image_id"]] = {
                "cls_id": name2id[annot["label"]],
                "bbox": xywh2yolo(x, y, width, height, imw, imh),
            }

        # Write annotations for each image
        for img_id, img_data in images.items():
            annot = annotations[img_id]

            # Copy image
            src_img_path = subset_img_src_dir / img_data["file_name"]
            shutil.copy(
                str(src_img_path), str(subset_img_dst_dir / img_data["file_name"])
            )

            output_txt_file = (
                subset_annot_dst_dir / img_data["file_name"]
            ).with_suffix(".txt")

            with output_txt_file.open("w") as f:
                f.write(
                    f"{annot['cls_id']} {annot['bbox'][0]} {annot['bbox'][1]} {annot['bbox'][2]} {annot['bbox'][3]}\n"
                )

    # Create data.yaml file
    data_yml = {}
    for subset in annot_data["subsets"]:
        # map to new subset if provided
        new_subset = subset
        if subset_map and subset in subset_map:
            new_subset = subset_map[subset]

        data_yml[new_subset] = f"./images/{new_subset}"

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

    subset_map = {}
    for _map in args.subset_map:
        k, v = _map.split(":")
        subset_map[k] = v

    print(subset_map)

    convert_cvat_to_yolo_ultralytics(
        src_dir=args.src,
        output_dir=args.output,
        force=args.force,
        subset_map=subset_map,
    )


if __name__ == "__main__":
    main()
