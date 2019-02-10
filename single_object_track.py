import numpy as np
import cv2
import argparse
from collections import deque

HSV_RANGES = {
    'blue': (np.array([110, 100, 100]), np.array([130, 255, 255])),
    'green': (np.array([50, 100, 100]), np.array([70, 255, 255])),
    'red': (np.array([170, 100, 100]), np.array([190, 255, 255])),
}

KERNEL_SIZE = 5

def main():
    cap = cv2.VideoCapture(0)

    # keep the motion of the center of detected objects
    center_path = deque(maxlen=64)

    while True:
        ret, img = cap.read()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # detect objects with color bule
        mask = generate_mask(hsv, 'blue')
        #mask_red = generate_mask(hsv, 'red')
        #mask = cv2.bitwise_or(mask_blue, mask_red)

        # filter the image with the mask
        res = cv2.bitwise_and(img, img, mask=mask)

        # Find two outmost contours(boundaries) for the detected objects
        # RETR_EXTERNAL: only find the outmost and ignore all the child contours
        # CHAIN_APPROX_SIMPLE: only keep the few kep center_path of the contours
        contours, _ = cv2.findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[-2:]
        center = None

        if len(contours) > 0:
            # Find the contour with largest area
            c = max(contours, key=cv2.contourArea)

            # The minimal enclosing circle for that contour
            ((x, y), radius) = cv2.minEnclosingCircle(c)

            # Use the moments to calculate the center
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            if radius > 5:
                # draw a yellow circle the thickness 2
                cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)

                # draw a red filled cicle at the center
                cv2.circle(img, center, 5, (0, 0, 255), -1)

        center_path.appendleft(center)
        for i in range(1, len(center_path)):
            if center_path[i - 1] is None or center_path[i] is None:
                continue
            #thickness decays with sqrt of time
            thick = int(np.sqrt(len(center_path) / float(i + 1)) * 2.5)
            #plot a red line between each point
            cv2.line(img, center_path[i - 1], center_path[i], (0, 0, 225), thick)

        cv2.imshow("Frame", img)
        cv2.imshow("mask", mask)
        cv2.imshow("res", res)

        # show each image for 30 ms
        k = cv2.waitKey(30) & 0xFF
        # if 'Space' is pressed, break the loop
        if k == 32:
            break

    # cleanup the camera and close any open windows
    cap.release()
    cv2.destroyAllWindows()

def generate_mask(img: np.ndarray, color: str):
    """
    Args:
        img: images with HSV values
        color: the color of the foreground object
    """

    mask = cv2.inRange(img, *HSV_RANGES[color])
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)

    # Erosion: Use the kernel to slide through the image (as in 2D convolution),
    # A pixel in the original image (either 1 or 0) will be considered 1 only if all
    # the pixels under the kernel is 1, otherwise it is eroded (made to zero).
    # Useful for remove white noise, but also shrinks the object. Usually followed
    # by a dilation.
    mask = cv2.erode(mask, kernel, iterations=2)

    # Opening: An erosion followed by dilation. Useful to remove white noise.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Dilation: It is just opposite of erosion. A pixel element is ‘1’ if at least
    # one pixel under the kernel is ‘1’. Makes the objects bigger.
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Closing: A dilation followed by an erosion. Useful to close small holes in
    # the foreground objects.
    #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

if __name__ == '__main__':
    main()
