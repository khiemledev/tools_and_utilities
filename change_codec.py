import cv2
import subprocess
import sys
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print('Usage: python change_codec.py <input_video_file>')
        sys.exit(1)
    path = Path(sys.argv[1])

    # Input and output file paths
    input_video_file = str(path)
    output_video_file = path.stem + '_' + '.mp4'

    # FFmpeg command to encode video with H.264 codec and audio with AAC codec
    ffmpeg_cmd = (
        f'ffmpeg -y -i "{input_video_file}" '
        '-c:v libx264 -preset slow -crf 22 '  # H.264 video settings
        '-c:a aac -b:a 128k '                # AAC audio settings
        f'"{output_video_file}"'
    )

    # Use OpenCV to get video information (fps, frame size, etc.)
    video_cap = cv2.VideoCapture(input_video_file)
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Execute the FFmpeg command
    subprocess.run(ffmpeg_cmd, shell=True)

    # Release the video capture object
    video_cap.release()


if __name__ == "__main__":
    main()
