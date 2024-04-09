"""
This script will read video and stream it to HLS.
Author: Khiem Le
Github: https://github.com/khiemledev
Date: 2024-04-09
"""

import time
from argparse import ArgumentParser
from pathlib import Path

import av
import cv2


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--video-path",
        type=str,
        help="Path to video file",
        required=True,
    )
    parser.add_argument(
        "--hls-output-path",
        type=str,
        default="./hls_serving_file/hls.m3u8",
        help="Output of the HLS stream",
    )
    parser.add_argument(
        "--frame-size",
        type=int,
        nargs=2,
        default=(640, 480),
        help="Size of the frames in the HLS stream. Defaults to (640, 480)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Frame rate of the HLS stream. Defaults to 25",
    )
    parser.add_argument(
        "--segment-duration",
        type=int,
        default=15,
        help="Duration of each segment in the HLS stream. Defaults to 15",
    )
    return parser.parse_args()


def main():
    args = get_args()

    video_path = args.video_path

    hls_path = Path(args.hls_output_path)
    hls_path.parent.mkdir(parents=True, exist_ok=True)

    frame_size = args.frame_size

    segment_duration = args.segment_duration

    fps = args.fps

    input_container = av.open(video_path)

    # Create the output container for HLS
    output_container = av.open(
        str(hls_path),
        "w",
        format="hls",
    )

    # Define the video stream in the output container
    output_stream = output_container.add_stream(
        "libx264", rate=fps
    )  # Adjust the output codec and frame rate as needed

    last_time = 0
    tic = time.time()
    for frame in input_container.decode(video=0):
        toc = time.time()
        recorded_time = toc - tic

        # Perform any necessary frame processing here
        # For example, you can modify the frame using OpenCV operations

        output_frame = frame.to_ndarray(format="bgr24")
        output_frame = cv2.resize(output_frame, frame_size)

        # Encode the modified frame
        # output_frame = cv2.cvtColor(frame.to_image(), cv2.COLOR_RGB2BGR)
        # output_frame = cv2.flip(output_frame, 0)  # Example processing: flip vertically

        # Convert the frame to AVFrame format
        output_av_frame = av.VideoFrame.from_ndarray(output_frame, format="bgr24")

        # # Set the output frame parameters
        # output_av_frame.width = output_stream.width
        # output_av_frame.height = output_stream.height

        # Encode and write the frame to the output container
        for packet in output_stream.encode(output_av_frame):
            output_container.mux(packet)

        # Segment the output HLS stream based on the segment duration
        # print(recorded_time, last_time)
        if recorded_time - last_time >= segment_duration:
            last_time = recorded_time

            output_container.close()

            output_container = av.open(str(hls_path), "w", format="hls")
            output_stream = output_container.add_stream(
                "libx264", rate=fps
            )  # Adjust as needed
            print("Segmented!")

        # time.sleep(wait_ms / 1000)

    # Close the input and output containers
    input_container.close()
    output_container.close()

    print("Done!")


if __name__ == "__main__":
    main()
