import cv2
from pathlib import Path


def main():
    frames_dir = Path("../StreamingVideo/video_frames")
    output_path = Path("./output.mp4")

    frame_size = (640, 480)
    fps = 5

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)

    for frame_path in sorted(frames_dir.glob("*.jpg")):
        frame = cv2.imread(str(frame_path))
        frame = cv2.resize(frame, frame_size)
        out.write(frame)

    # Release everything if job is finished
    out.release()


if __name__ == "__main__":
    main()
