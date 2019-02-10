import cv2
import numpy as np


def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, img = cap.read()
        img = cv2.flip(img, 1)

        # select only the skin color
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 50, 70]), np.array([25, 150, 255]))
        res = cv2.bitwise_and(img, img, mask=mask)

        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        median = cv2.GaussianBlur(gray, (5, 5), 0)
        kernel_square = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(median, kernel_square, iterations=2)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel_square)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_square)
        ret, thresh = cv2.threshold(opening, 30, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )

        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 2500:
                x, y, w1, h1 = cv2.boundingRect(contour)
                cv2.drawContours(img, contour, -1, (0, 255, 0), 3)
                newImage = thresh[y : y + h1, x : x + w1]
                newImage = cv2.resize(newImage, (50, 50))

        cv2.imshow("Frame", img)
        cv2.imshow("Contours", newImage)

        # show each image for 10 ms
        k = cv2.waitKey(10) & 0xFF
        # if 'Space' is pressed, break the loop
        if k == 32:
            break


if __name__ == "__main__":
    main()
