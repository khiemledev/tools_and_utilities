import cv2
from pathlib import Path


def main():
    video_path = Path('../StreamingVideo/live.mp4')
    output_dir = Path('video_frames/')

    skip_frame = 0 # 0 mean no skip
    max_frames = 220 # 0 mean no limit

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
        if max_frames > 0 and frame_count > max_frames:
            break

        # Skip frame if needed
        if skip_frame > 0 and frame_idx % skip_frame != 0:
            continue

        # Save frame to output directory
        cv2.imwrite(str(output_dir / f'{frame_idx:06d}.jpg'), frame)

        frame_count += 1

    # Release video capture object
    cap.release()


if __name__ == "__main__":
    main()
