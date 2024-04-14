import xml.etree.ElementTree as ET
from pathlib import Path

StrPath = str | Path


def read_cvat_annotation_xml(xml_path: StrPath) -> dict:
    """Read

    Args:
        xml_path (StrPath): path to the annotation xml file in CVAT dataset folder

    Raises:
        ValueError: value error with explaination

    Returns:
        dict: dictionary containing all the data read from the CVAT XML file
    """  # noqa: E501

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
                    "subset": img_subset,
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
