"""
This script combine multiple frames into a video.
Author: Khiem Le
Github: https://github.com/khiemledev
Date: 2024-04-09
"""

from argparse import ArgumentParser
from ast import parse
from pathlib import Path

import cv2


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--frame-dir",
        type=str,
        help="Path to the directory containing the frames",
        required=True,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to the output video file",
        required=True,
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="jpg",
        help="Extension of the frames. Defaults to jpg",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Frames per second. Defaults to 5",
    )
    parser.add_argument(
        "--frame-size",
        type=int,
        nargs=2,
        default=(640, 480),
        help="Frame size. Defaults to (640, 480)",
    )
    return parser.parse_args()


def main():
    args = get_args()
    print(args.frame_size)

    frames_dir = Path(args.frame_dir)
    output_path = Path(args.output_path)

    if output_path.exists():
        print("!!! Output file already exists")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_size = args.frame_size
    fps = args.fps

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)

    for frame_path in sorted(frames_dir.glob(f"*.{args.ext}")):
        frame = cv2.imread(str(frame_path))
        frame = cv2.resize(frame, frame_size)
        out.write(frame)

    # Release everything if job is finished
    out.release()

    print("Done!")


if __name__ == "__main__":
    main()
