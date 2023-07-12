import cv2
import time

def main():
    video_url = 'https://streaming.url/index.m3u8'

    live_output = 'live.mp4'
    frame_size = (1280, 720)

    record_time = 60 * 30 # in seconds, -1 for infinite

    cap = cv2.VideoCapture(video_url)
    if (cap.isOpened() == False):
        print('!!! Unable to open URL')
        return

    # calculate FPS using formulate frames / seconds
    tic = time.time()
    toc = time.time()
    n_frames = 0
    print('Calculating FPS...')
    while cap.isOpened() and toc - tic < 10:
        toc = time.time()
        ret, frame = cap.read()
        if not ret:
            print('!!! Failed to read frame')
            break

        n_frames += 1

    fps = round(n_frames / (toc - tic))

    # retrieve FPS and calculate how long to wait between each frame to be display
    # fps = cap.get(cv2.CAP_PROP_FPS)
    wait_ms = int(1000 / fps)
    print('FPS:', fps)

    fourcc = cv2.VideoWriter_fourcc(*('MJPG' if live_output.endswith('avi') else 'mp4v'))
    output_writer = cv2.VideoWriter(live_output, fourcc, fps, frame_size)  # Adjust the frame size if needed

    tic = time.time()
    try:
        toc = time.time()
        recorded_time = toc - tic

        last_time = 0

        print('Recording...')
        while cap.isOpened() and (record_time < 0 or recorded_time < record_time):
            toc = time.time()
            recorded_time = toc - tic

            if recorded_time - last_time > 60:
                print('Recorded time:', recorded_time)
                last_time = recorded_time

            # read one frame
            ret, frame = cap.read()
            if not ret:
                print('!!! Failed to read frame')
                break

            # TODO: perform frame processing here
            frame = cv2.resize(frame, frame_size)

            # Write frame to the output video file
            output_writer.write(frame)
            # cv2.imwrite('frame.jpg', frame)

            # display frame
            # cv2.imshow('frame',frame)
            # print('Read a new frame: ', ret)
            # if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            #     break


    except KeyboardInterrupt:
        pass

    cap.release()
    output_writer.release()
    # # cv2.destroyAllWindows()
    # # Finish writing the HLS segments and playlists


if __name__ == "__main__":
    main()
