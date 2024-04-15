from pathlib import Path

import yaml

StrPath = str | Path

SUPPORTED_IMG_EXTS = ("jpg", "jpeg", "png")


def read_data_yaml(path: StrPath) -> dict:
    """Read data inside data.yaml file and return a dictionary

    Args:
        path (StrPath): path to data.yaml

    Raises:
        ValueError: description about the error

    Returns:
        dict: data inside data.yaml
    """

    path = Path(path)

    with path.open("r") as f:
        data = yaml.safe_load(f)

    if not data:
        raise ValueError(f"data.yml is empty: {data}")

    names = data.get("names")
    if names is None:
        raise ValueError(f"data.yml does not have 'names': {data}")

    # Check if names in data_yml is list or dict. If dict, convert it to list
    if isinstance(data, dict):
        _names = list(data["names"].items())
        _names = sorted(_names, key=lambda x: x[0])  # sort by key - class id
        names = [x[1] for x in _names]

    data["names"] = names

    if "nc" not in data or data.get("nc") is None:
        data["nc"] = len(data["names"])
    elif int(data["nc"]) != len(names):
        raise ValueError("data.yml does not have correct number of 'names': {data}")

    return data


def validate_dataset_folder(data_yml: dict, root_dir: StrPath) -> dict:
    """_summary_

    Args:
        data_yml (dict): data inside yaml file
        root_dir (StrPath): path to dataset directory

    Raises:
        ValueError: description about the error
    Returns:
        dict: {
            "<subset>": [
                {
                    "image": <path_to_image>,
                    "label": [
                        [cls_id, xcn, ycn, bwn, bhn],
                        ...
                    ],
                },
                ...
            ],
            ...
        }
    """

    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise ValueError(f"Root directory does not exist: {root_dir}")

    if not root_dir.is_dir():
        raise ValueError(f"Root directory is not a directory: {root_dir}")

    metadata_attrs = set(("nc", "names"))
    allowed_subsets = set(("train", "val", "test"))
    subsets = set(data_yml.keys()) - metadata_attrs

    # Dataset set must contains 'train' and 'val' key, 'test' is optional
    if "train" not in subsets:
        raise ValueError("Dataset set must contains 'train' and 'val' key")
    if "val" not in subsets:
        raise ValueError("Dataset set must contains 'train' and 'val' key")

    # Any other subset is not allowed
    if len(subsets - allowed_subsets) > 0:
        raise ValueError(
            f"Dataset set contains invalid subsets: {subsets - allowed_subsets}"
        )

    result = {}

    # There are two kind of data in data.yaml file.
    # First, the subset key contains relative path to subset images directory
    # Second, the subset key contains relative path to the file which contains
    # relative paths to subset image path
    for subset in subsets:
        if subset not in result:
            result[subset] = []

        if data_yml[subset].endswith(".txt"):
            # Second type
            print("Second type of data.yaml")

            subset_txt = Path(data_yml[subset])
            if subset_txt.is_absolute():
                raise ValueError(f"Subset txt path is absolute: {subset_txt}")

            subset_txt = root_dir / subset_txt

            if not subset_txt.exists():
                raise ValueError(f"Subset txt path does not exist: {subset_txt}")

            if not subset_txt.is_file():
                raise ValueError(f"Subset txt path is not a file: {subset_txt}")

            with subset_txt.open("r") as f:
                img_paths = [e.strip() for e in f.readlines()]
                if len(img_paths) == 0:
                    raise ValueError(f"Subset txt is empty: {subset_txt}")

                test_path = Path(img_paths[0])
                if test_path.is_absolute():
                    raise ValueError(
                        f"Image path inside {subset_txt} is absolute: {test_path}"
                    )

                test_path = root_dir / test_path
                if test_path.exists() is False:
                    raise ValueError(f"Image path does not exist: {test_path}")

                if not test_path.is_file():
                    raise ValueError(f"Image path is not a file: {test_path}")

                # For each image, check label in txt file and append labels to result
                for img_path in img_paths:
                    img_path = root_dir / img_path

                    txt_path = Path(
                        str(img_path).replace("/images/", "/labels/")
                    ).with_suffix(".txt")
                    if not txt_path.exists():
                        raise ValueError(f"Label file {txt_path} not found")

                    with txt_path.open("r") as f:
                        labels = [l.strip().split(" ") for l in f if l.strip() != ""]

                        # cls_id is int, not float
                        for i in range(len(labels)):
                            labels[i] = list(map(float, labels[i]))
                            labels[i][0] = int(labels[i][0])

                        result[subset].append(
                            {
                                "image": str(img_path),
                                "labels": labels,
                            }
                        )
        else:
            # First type
            print("First type of data.yaml")

            subset_imgs_path = Path(data_yml[subset])
            if subset_imgs_path.is_absolute():
                raise ValueError(f"Subset images path is absolute: {subset_imgs_path}")

            subset_imgs_path = root_dir / subset_imgs_path
            if not subset_imgs_path.exists():
                raise ValueError(
                    f"Subset images path does not exist: {subset_imgs_path}"
                )

            if not subset_imgs_path.is_dir():
                raise ValueError(
                    f"Subset images path is not a directory: {subset_imgs_path}"
                )

            # List all images inside images directory
            img_paths = []
            for ext in SUPPORTED_IMG_EXTS:
                _img_paths = list(subset_imgs_path.glob(f"*.{ext}"))
                img_paths.extend(_img_paths)

            if len(img_paths) == 0:
                raise ValueError(f"No image found in subset {subset}")

            # For each image, check label in txt file and append labels to result
            for img_path in img_paths:
                txt_path = Path(
                    str(img_path).replace("/images/", "/labels/")
                ).with_suffix(".txt")
                if not txt_path.exists():
                    raise ValueError(f"Label file {txt_path} not found")

                with txt_path.open("r") as f:
                    labels = [l.strip().split(" ") for l in f if l.strip() != ""]

                    # cls_id is int, not float
                    for i in range(len(labels)):
                        labels[i] = list(map(float, labels[i]))
                        labels[i][0] = int(labels[i][0])

                    result[subset].append(
                        {
                            "image": str(img_path),
                            "labels": labels,
                        }
                    )

    return result
