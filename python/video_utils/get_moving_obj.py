"""
This script will get moving objects by comparing frames with background image.
Author: Khiem Le
Github: https://github.com/khiemledev
Date: 2024-04-09
"""

from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--bg-path",
        type=str,
        help="Path to the background image",
        required=True,
    )
    parser.add_argument(
        "--frame-dir",
        type=str,
        help="Path to the directory containing the frames",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to the output directory",
        required=True,
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="jpg",
        help="Extension of the frames. Defaults to jpg",
    )
    parser.add_argument(
        "--resize-img",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Resize the frame size before processing. Defaults to (256, 256)",
    )
    parser.add_argument(
        "--morphological-kernel-size",
        type=int,
        nargs=2,
        default=(3, 3),
        help="morphological_kernel_size. Defaults to (3, 3). "
        "If resize to smaller image, use smaller kernel size",
    )
    parser.add_argument(
        "--min-contour-area",
        type=int,
        default=50,
        help="min_contour_area. Defaults to 50",
    )
    parser.add_argument(
        "--min-bin-thres",
        type=int,
        default=13,
        help="min_bin_thres. Defaults to 13",
    )
    parser.add_argument(
        "--max-bin-thres",
        type=int,
        default=13,
        help="max_bin_thres. Defaults to 256",
    )
    parser.add_argument(
        "--stack-orin-image",
        type=bool,
        default=True,
        help="stack_orin_image. Defaults to True",
    )
    parser.add_argument(
        "--stack-type",
        type=str,
        default="horizontal",
        help="stack_type. Defaults to horizontal",
        choices=["horizontal", "vertical"],
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.1,
        help="scale_factor. Defaults to 1.1",
    )
    return parser.parse_args()


def main():
    args = get_args()

    resize_img = args.resize_img
    morphological_kernel_size = args.morphological_kernel_size
    min_contour_area = args.min_contour_area

    min_bin_thres = args.min_bin_thres
    max_bin_thres = args.max_bin_thres

    stack_orin_image = args.stack_orin_image
    stack_type = args.stack_type

    scale_factor = args.scale_factor

    color_channel = None  # None to disable, remember to change min_bin_thres

    background_image = cv2.imread(args.bg_path)
    if resize_img:
        background_image = cv2.resize(background_image, resize_img)

    if color_channel:
        background_image = cv2.cvtColor(background_image, color_channel)

    images_dir = Path(args.frame_dir)
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Remove all file in output directory
    for file in save_dir.glob("*"):
        file.unlink()

    # # Create the background subtractor
    # bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    # Iterate through the frames
    for img_path in sorted(images_dir.glob(f"*.{args.ext}")):
        # Read image
        frame = cv2.imread(str(img_path))

        orin_frame = frame.copy()
        output_frame = frame.copy()

        # Check if background image is the same size as the frame
        if not resize_img and background_image.shape != frame.shape:
            raise ValueError("Background image size does not match frame size")

        if resize_img:
            frame = cv2.resize(frame, resize_img)

        if color_channel:
            frame = cv2.cvtColor(frame, color_channel)

        # Apply background subtraction using the background image
        # fg_mask = bg_subtractor.apply(frame, learningRate=0)

        # # Threshold the foreground mask
        # _, binary_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

        # Compute the absolute difference between the frame and the background image
        diff = cv2.absdiff(frame, background_image)

        # Convert the difference image to grayscale
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) * scale_factor
        gray = np.clip(gray, 0, 255).astype(np.uint8)

        # Apply thresholding to obtain the binary image
        _, binary_mask = cv2.threshold(
            gray, min_bin_thres, max_bin_thres, cv2.THRESH_BINARY
        )

        # Perform morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morphological_kernel_size)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        # # Find contours of moving objects
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # # Iterate through contours and filter small ones
        for contour in contours:
            if cv2.contourArea(contour) > min_contour_area:
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Scale the bounding box back to the original frame size
                x = int(x * orin_frame.shape[1] / frame.shape[1])
                y = int(y * orin_frame.shape[0] / frame.shape[0])
                w = int(w * orin_frame.shape[1] / frame.shape[1])
                h = int(h * orin_frame.shape[0] / frame.shape[0])

                cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Merge contours into a single region
        merged_mask = np.zeros_like(binary_mask)
        cv2.drawContours(merged_mask, contours, -1, (255), thickness=cv2.FILLED)

        # Stack frame and the mask
        if stack_orin_image:
            binary_mask = cv2.resize(merged_mask, orin_frame.shape[:2][::-1])
            if stack_type == "horizontal":
                merged_mask = np.hstack(
                    (output_frame, cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR))
                )
                stack2 = np.hstack(
                    (
                        orin_frame,
                        cv2.cvtColor(
                            cv2.resize(gray, orin_frame.shape[:2][::-1]),
                            cv2.COLOR_GRAY2BGR,
                        ),
                    )
                )
                merged_mask = np.vstack((stack2, merged_mask))
            elif stack_type == "vertical":
                merged_mask = np.vstack(
                    (orin_frame, cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR))
                )
            else:
                raise ValueError("Invalid stack type")

        # Save the resulting frame
        output = merged_mask
        cv2.imwrite(str(save_dir / img_path.name), output)

    print("Done!")


if __name__ == "__main__":
    main()
