from pathlib import Path
from pprint import pprint

import imagesize
import yaml


def read_data_yaml(path: Path) -> dict:
    """Read data inside data.yaml file and return a dictionary

    Args:
        path (Path): path to data.yaml

    Raises:
        ValueError: description about the error

    Returns:
        dict: data inside data.yaml

        {
            "nc": int - number of classes,
            "names": list[str] - list of class names
            "<subset>": str
        }
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


def read_imagenet(dataset_dir: Path) -> dict:
    """Read data inside imagenet folder and return a dictionary

    Args:
        dataset_dir (Path): path to imagenet folder

    Returns:
        dict: data inside imagenet folder

        {
            "nc": <number of classes>,
            "names": <list of class names>,
            "<subset>: [
                {
                    "file_path": <file_path>,
                    "filename": <filename>,
                    "width": <width>,
                    "height": <height>,
                    "label": <label>,
                    "id": <id>,
                },
                ...
            ],
        }
    """

    result = {}

    if not dataset_dir.exists():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")

    if not dataset_dir.is_dir():
        raise ValueError(f"Dataset is not a directory: {dataset_dir}")

    class_names: list | None = None
    subsets = set()

    # read data.yaml file if exist
    data_yaml_path = dataset_dir / "data.yaml"
    if data_yaml_path.exists():
        data_yaml = read_data_yaml(data_yaml_path)
        pprint(data_yaml)

        class_names = data_yaml["names"]
        subsets = set(data_yaml.keys())
        subsets.remove("names")
        subsets.remove("nc")

    pprint(class_names)
    pprint(subsets)

    # List all directories inside dataset_dir, which is subsets
    for subset in dataset_dir.iterdir():
        if not subset.is_dir():
            if subset.name not in ["data.yaml", "data.yml"]:
                raise ValueError(f"Dataset is not a directory: {subset}")
            else:
                # skip data.yaml file
                continue

        subsets.add(subset.name)

        _names = list()
        for label in subset.iterdir():
            if not label.is_dir():
                raise ValueError(f"Dataset is not a directory: {label}")

            _names.append(label.name)

        if class_names is None:
            class_names = _names
        else:
            # if class_names is not none, check if names of each directory are the same
            if set(class_names).difference(set(_names)):
                raise ValueError(f"Class names are not the same: {class_names} != {_names}")

        idx = 0
        imgs = []

        # Start to read all images inside each subset/label folder
        for label_dir in subset.iterdir():
            img_paths = label_dir.iterdir()

            for img_path in img_paths:
                if not img_path.is_file():
                    raise ValueError(f"Dataset is not a file: {img_path}")

                width, height = imagesize.get(img_path.as_posix())

                imgs.append(
                    {
                        "file_path": img_path.as_posix(),
                        "filename": img_path.name,
                        "height": height,
                        "width": width,
                        "label": label_dir.name,
                        "id": idx,
                    }
                )
                idx += 1

        result[subset.name] = imgs

    result["names"] = list(class_names or set())
    result["nc"] = len(result["names"])
    result["subsets"] = subsets

    return result
