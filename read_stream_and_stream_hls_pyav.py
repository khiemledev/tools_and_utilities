import av
import cv2
import time
from pathlib import Path


def main():
    video_url = 'https://example.com/index.m3u8'

    live_output = 'output123123.mp4'
    hls_path = Path('./hls_serving_file/hls.m3u8')
    hls_path.parent.mkdir(parents=True, exist_ok=True)

    frame_size = (640, 480)
    record_time = -1 # in seconds, -1 for infinite

    segment_duration = 60

    fps = 23

    input_container = av.open(video_url)

    # Create the output container for HLS
    output_container = av.open(
        str(hls_path),
        'w',
        format='hls',
    )

    # Define the video stream in the output container
    output_stream = output_container.add_stream('libx264', rate=fps)  # Adjust the output codec and frame rate as needed

    last_time = 0
    tic = time.time()
    for frame in input_container.decode(video=0):
        toc = time.time()
        recorded_time = toc - tic

        # Perform any necessary frame processing here
        # For example, you can modify the frame using OpenCV operations

        output_frame = frame.to_ndarray(format='bgr24')
        output_frame = cv2.resize(output_frame, frame_size)

        # Encode the modified frame
        # output_frame = cv2.cvtColor(frame.to_image(), cv2.COLOR_RGB2BGR)
        # output_frame = cv2.flip(output_frame, 0)  # Example processing: flip vertically

        # Convert the frame to AVFrame format
        output_av_frame = av.VideoFrame.from_ndarray(output_frame, format='bgr24')

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

            output_container = av.open(str(hls_path), 'w', format='hls')
            output_stream = output_container.add_stream('libx264', rate=fps)  # Adjust as needed
            print('Segmented!')

        # time.sleep(wait_ms / 1000)


    # Close the input and output containers
    input_container.close()
    output_container.close()
    print('Done!')


if __name__ == "__main__":
    main()
