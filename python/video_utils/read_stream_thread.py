import time
from threading import Thread

import cv2


class ThreadedCamera(object):
    def __init__(
        self,
        src=0,
        buffer_size=2,
        fps=30,
        resized_width=640,
    ):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)

        self.orin_size = (
            int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self.resized_size = (
            resized_width,
            int(resized_width * self.orin_size[1] / self.orin_size[0]),
        )
        print("Resized size: ", self.resized_size)

        self.FPS = fps
        self.WAIT_MS = int(1000 / self.FPS)

        # Start frame retrieval thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()

                self.frame = cv2.resize(self.frame, self.resized_size)

            time.sleep(self.WAIT_MS / 1000.0)

    def show_frame(self):
        cv2.imshow("frame", self.frame)
        if cv2.waitKey(self.WAIT_MS) & 0xFF == ord("q"):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(0)


def main():
    src = "source_video"
    threaded_camera = ThreadedCamera(src)
    while True:
        try:
            threaded_camera.show_frame()
        except AttributeError:
            pass


if __name__ == "__main__":
    main()
