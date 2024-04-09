"""
This script will get the background images by getting the mean of the frames
pixel value.
Author: Khiem Le
Github: https://github.com/khiemledev
Date: 2024-04-09
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--frame-dir",
        type=str,
        help="Path to the directory containing the frames",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to the output image file",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="jpg",
        help="Extension of the frames. Defaults to jpg",
    )
    return parser.parse_args()


def main():
    args = get_args()

    images_dir = Path(args.frame_dir)
    output_path = Path(args.output_path)
    if output_path.exist():
        print("!!! Output file already exists")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for img_path in sorted(images_dir.glob(f"*.{args.ext}")):
        img = Image.open(img_path)
        frames.append(np.array(img))

    # Stack the 3 images into a 4d sequence
    sequence = np.stack(frames, axis=3)

    # Repace each pixel by mean of the sequence
    result = np.median(sequence, axis=3).astype(np.uint8)

    # Save to disk
    Image.fromarray(result).save(output_path)

    print("Done!")


if __name__ == "__main__":
    main()
