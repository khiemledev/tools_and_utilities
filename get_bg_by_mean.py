import cv2
import numpy as np

from PIL import Image
from pathlib import Path


def main():
    images_dir = Path('./video_frames/')
    output_path = Path('bg.jpg')

    frames = []
    for img_path in sorted(images_dir.glob('*.jpg')):
        img = Image.open(img_path)
        frames.append(np.array(img))

    # Stack the 3 images into a 4d sequence
    sequence = np.stack(frames, axis=3)

    # Repace each pixel by mean of the sequence
    result = np.median(sequence, axis=3).astype(np.uint8)

    # Save to disk
    Image.fromarray(result).save(output_path)


if __name__ == "__main__":
    main()
