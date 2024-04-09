"""
This script will read stream from url and stream it to HLS.
Author: Khiem Le
Github: https://github.com/khiemledev
Date: 2024-04-09
"""

import time
from argparse import ArgumentParser
from threading import Thread

import cv2


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--video-url",
        type=str,
        help="URL to the video stream",
        required=True,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Output video file",
        required=True,
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
        "--video-duration",
        type=int,
        default=30,
        help="Duration of recorded video. Default to 30 seconds",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Whether to display the frames in a window. Defaults to False",
    )
    return parser.parse_args()


def main():
    args = get_args()
    video_url = args.video_url

    output_path = args.output_path
    frame_size = args.frame_size

    video_duration = args.video_duration  # in seconds, -1 for infinite

    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        print("!!! Unable to open URL")
        return

    # calculate FPS using formulate frames / seconds
    tic = time.time()
    toc = time.time()
    n_frames = 0
    print("Calculating FPS...")
    while cap.isOpened() and toc - tic < 10:
        toc = time.time()
        ret, frame = cap.read()
        if not ret:
            print("!!! Failed to read frame")
            break

        n_frames += 1

    fps = round(n_frames / (toc - tic))

    # retrieve FPS and calculate how long to wait between each frame to be display
    # fps = cap.get(cv2.CAP_PROP_FPS)
    wait_ms = int(1000 / fps)
    print("FPS:", fps)

    fourcc = cv2.VideoWriter_fourcc(
        *("MJPG" if output_path.endswith("avi") else "mp4v")
    )
    output_writer = cv2.VideoWriter(
        output_path, fourcc, fps, frame_size
    )  # Adjust the frame size if needed

    tic = time.time()
    try:
        toc = time.time()
        recorded_time = toc - tic

        last_time = 0

        print("Recording...")
        while cap.isOpened() and (video_duration < 0 or recorded_time < video_duration):
            toc = time.time()
            recorded_time = toc - tic

            # Log every 60 seconds
            if recorded_time - last_time > 1:
                print("Recorded time:", recorded_time)
                last_time = recorded_time

            # read one frame
            ret, frame = cap.read()
            if not ret:
                print("!!! Failed to read frame")
                break

            # TODO: perform frame processing here
            frame = cv2.resize(frame, frame_size)

            # Write frame to the output video file
            output_writer.write(frame)

            # display frame
            if args.display:
                cv2.imshow("frame", frame)
                print("Read a new frame: ", ret)
                if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(wait_ms / 1000.0)

    except KeyboardInterrupt:
        pass

    cap.release()
    output_writer.release()

    if args.display:
        cv2.destroyAllWindows()

    print("Done!")


if __name__ == "__main__":
    main()
