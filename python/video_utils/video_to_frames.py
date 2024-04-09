"""
This script split video into frames.
Author: Khiem Le
Github: https://github.com/khiemledev
Date: 2024-04-09
"""

from argparse import ArgumentParser
from pathlib import Path

import cv2


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--video-path",
        type=str,
        help="Path to the video file",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to the output directory",
        required=True,
    )
    parser.add_argument(
        "--skip-frame",
        type=int,
        default=5,
        help="Number of frames to skip. Defaults to 0",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Maximum number of frames to process. Defaults to -1 (no set)",
    )
    return parser.parse_args()


def main():
    args = get_args()

    video_path = Path(args.video_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove all file in output directory
    for file in output_dir.glob("*"):
        file.unlink()

    skip_frame = args.skip_frame  # 0 mean no skip
    max_frames = args.max_frames  # 0 mean no limit

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize video capture object
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Error opening video file")
        return

    # Read until video is completed
    frame_idx = 0
    frame_count = 0
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame_idx += 1

        # Stop if max_frames is reached
        if max_frames != -1 and frame_count > max_frames:
            break

        # Skip frame if needed
        if skip_frame > 0 and frame_idx % skip_frame != 0:
            continue

        # Save frame to output directory
        cv2.imwrite(str(output_dir / f"{frame_idx:06d}.jpg"), frame)

        frame_count += 1

    # Release video capture object
    cap.release()


if __name__ == "__main__":
    main()
    main()
    main()
