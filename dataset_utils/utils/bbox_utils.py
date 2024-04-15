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


def yolo2xyxy(
    xc: int,
    yc: int,
    box_w: int,
    box_h: int,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    x1 = (xc - box_w / 2) * img_w
    y1 = (yc - box_h / 2) * img_h
    x2 = (xc + box_w / 2) * img_w
    y2 = (yc + box_h / 2) * img_h
    return x1, y1, x2, y2
