import json
from pathlib import Path

StrPath = str | Path


def read_coco_dataset(root: StrPath) -> dict:
    """Read COCO dataset

    Args:
        root (StrPath): path to dataset

    Raises:
        ValueError: description about the error
    Returns:
        dict: {
            "idx2name": {
                "1": "person",
                "2": "bicycle",
                ...
            },
            "<subset>": [
                {
                    "image": <path_to_image>,
                    "labels": [
                        [cls_id, x1, y1, w, h],
                        ...
                    ],
                },
                ...
            ],
            ...
        }
    """

    root = Path(root)
    if not root.exists():
        raise ValueError(f"COCO dataset {root} does not exist")

    images_dir = root / "images"
    annotations_dir = root / "annotations"

    if not images_dir.exists():
        raise ValueError("images folder not found")

    if not annotations_dir.exists():
        raise ValueError("annotations folder not found")

    annot_files = list(annotations_dir.glob("instances_*.json"))
    if len(annot_files) == 0:
        raise ValueError("annotations files not found")

    result = {}

    idx2name = None

    for annot_file in annot_files:
        subset = annot_file.stem.split("instances_")[-1]
        subset_data = []

        with annot_file.open() as f:
            data = json.load(f)

        categories = data["categories"]
        if idx2name is None:
            idx2name = {c["id"]: c["name"] for c in categories}

        images = {c["id"]: c for c in data["images"]}
        annotations = data["annotations"]

        # this loop assign labels to correct image with image_id
        for annot in annotations:
            """
                {
                "id": 86,
                "image_id": 19,
                "category_id": 2,
                "segmentation": [],
                "area": 672.4672000000002,
                "bbox": [
                    286.9,
                    231.92000000000002,
                    23.480000000000018,
                    28.639999999999986
                ],
                "iscrowd": 0
                },
            """
            img_id = annot["image_id"]
            if img_id not in images:
                print("image_id not exist")
                continue

            if "labels" not in images[img_id]:
                images[img_id]["labels"] = []

            images[img_id]["labels"].append(annot)

        for img_data in images.values():
            img_path = images_dir / img_data["file_name"]
            if not img_path.exists():
                raise ValueError(f"image {img_path} does not exist")

            # convert labels to format [cls_id, x1, y1, w, h]
            labels = []
            for annot in img_data["labels"]:
                cls_id = annot["category_id"]
                x1, y1, w, h = annot["bbox"]
                labels.append([cls_id, x1, y1, w, h])
            subset_data.append({"image": img_path.as_posix(), "labels": labels})

        result[subset] = subset_data

    result["idx2name"] = idx2name

    return result
