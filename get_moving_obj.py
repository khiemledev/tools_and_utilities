import cv2
import numpy as np

from pathlib import Path


def main():
    resize_img = (256, 256) # None to disable, or (height, width)
    morphological_kernel_size = (3, 3) # If resize to smaller image, use smaller kernel size
    min_contour_area = 50

    min_bin_thres = 13
    max_bin_thres = 256

    stack_orin_image = True
    stack_type = 'horizontal' # 'horizontal' or 'vertical

    scale_factor = 1.1

    color_channel = None # None to disable, remember to change min_bin_thres

    background_image = cv2.imread('bg.jpg')
    if resize_img:
        background_image = cv2.resize(background_image, resize_img)

    if color_channel:
        background_image = cv2.cvtColor(background_image, color_channel)

    images_dir = Path('./video_frames/')
    save_dir = Path('./video_moving_obj/')
    save_dir.mkdir(parents=True, exist_ok=True)

    # Remove all file in output directory
    for file in save_dir.glob('*'):
        file.unlink()

    # # Create the background subtractor
    # bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    # Iterate through the frames
    for img_path in sorted(images_dir.glob('*.jpg')):
        # Read image
        frame = cv2.imread(str(img_path))

        orin_frame = frame.copy()
        output_frame = frame.copy()

        # Check if background image is the same size as the frame
        if not resize_img and background_image.shape != frame.shape:
            raise ValueError('Background image size does not match frame size')

        if resize_img:
            frame = cv2.resize(frame, resize_img)

        if color_channel:
            frame = cv2.cvtColor(frame, color_channel)

        # Apply background subtraction using the background image
        # fg_mask = bg_subtractor.apply(frame, learningRate=0)

        # # Threshold the foreground mask
        # _, binary_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

         # Compute the absolute difference between the frame and the background image
        diff = cv2.absdiff(frame, background_image)

        # Convert the difference image to grayscale
        gray = (cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) * scale_factor)
        gray = np.clip(gray, 0, 255).astype(np.uint8)

        # Apply thresholding to obtain the binary image
        _, binary_mask = cv2.threshold(gray, min_bin_thres, max_bin_thres, cv2.THRESH_BINARY)

        # Perform morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morphological_kernel_size)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)


        # # Find contours of moving objects
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # # Iterate through contours and filter small ones
        for contour in contours:
            if cv2.contourArea(contour) > min_contour_area:
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Scale the bounding box back to the original frame size
                x = int(x * orin_frame.shape[1] / frame.shape[1])
                y = int(y * orin_frame.shape[0] / frame.shape[0])
                w = int(w * orin_frame.shape[1] / frame.shape[1])
                h = int(h * orin_frame.shape[0] / frame.shape[0])

                cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Merge contours into a single region
        merged_mask = np.zeros_like(binary_mask)
        cv2.drawContours(merged_mask, contours, -1, (255), thickness=cv2.FILLED)

        # Stack frame and the mask
        if stack_orin_image:
            binary_mask = cv2.resize(merged_mask, orin_frame.shape[:2][::-1])
            if stack_type == 'horizontal':
                merged_mask = np.hstack((output_frame, cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)))
                stack2 = np.hstack((orin_frame, cv2.cvtColor(cv2.resize(gray, orin_frame.shape[:2][::-1]), cv2.COLOR_GRAY2BGR)))
                merged_mask = np.vstack((stack2, merged_mask))
            elif stack_type == 'vertical':
                merged_mask = np.vstack((orin_frame, cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)))
            else:
                raise ValueError('Invalid stack type')

        # Save the resulting frame
        output = merged_mask
        cv2.imwrite(str(save_dir / img_path.name), output)


if __name__ == "__main__":
    main()
