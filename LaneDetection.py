# ====================================
# Lane Detection Program, version 1.5
# ====================================
#
# Name: Aldrei Chua
# Date: 3/12/2025
# Version: 1.5
#
# Description
# ------------
# Receives a video showing a car driving down a road and detects lane lines in the video and overlays them on top of the
# original video.
#
# Functions
# ----------
# detect_shapes() - Detects and overlays lane lines and arrows onto the image.
# preprocess() - Modifies the original image in order to improve detection.
# rotate_arrow() - Rotates the arrow PNG based on the value of the arrow_degree parameter.
#
# Algorithm
# ----------
# - Convert the image into bluescale
# - Apply a mask that isolates colors within a certain range of blue, which would ideally be the yellow and white lane
#   lines.
# - Convert the masked image into grayscale and blur it to reduce noise.
# - Dilate and erode the image to get rid of unwanted artifacts.
# - Apply a threshold mask that isolates strong, bright colors.
# - Dilate and erode the image again to remove artifacts.
# - Perform Canny Edge Detection to find strong lines in the frame.
# - Apply a mask that constricts the region of interest in order to only find lines ahead and to the sides of the car.
# - Use cv2.MORPH_GRADIENT to connect lines that may be broken.
# - Dilate and erode the image to remove artifacts.
# - Use Hough Transform to find the endpoints of the detected lines.
# - Use the slope-intercept formula to find the slopes and y-intercepts of each of the lines.
# - Separate the lines by if their slopes are positive or negative.
# - If the lines are too steep or too flat, they will not be included in the arrays.
# - If there is a positively-sloped line on the left side of the screen, it will not be included in the array.
# - If an array has no lines, draw the previous line detected to prevent flickering.
# - If the arrays have lines, use the slope and y-intercept of the median line of the sorted array and draw it on the
#   image.
# - Draw the centerline as the average of the two lane lines.
# - Perform Canny Edge Detection to find strong lines in the frame.
# - Apply a mask that constricts the region of interest in order to only find lines directly ahead of the car.
# - Dilate and erode the image to remove artifacts.
# - Use cv2.MORPH_GRADIENT to connect lines that may be broken.
# - Use cv2.findContours() to detect all the contours in the image.
# - Use cv2.approxPolyDP() to smooth out the contours and simplify its geometry.
# - Single out the contours with a specific length and area to prevent a random contour being detected instead of an
#   arrow.
# - Find the rectangle with the least area that can fit around the contour.
# - Single out the rectangles with a certain area.
# - Use the rectangles’ rotations to find out which way the arrow should turn.
# - Find rectangles that indicate that the car is moving straight again and set the arrow rotation back to the default
#   when those rectangles are detected.
# - Draw the bounding box around the arrow.
# - Draw the arrow contour detected.
# - Shrink the arrow PNG by a factor of 2.
# - Rotate the arrow based on the arrow_degree variable.
# - Add an alpha channel to the arrow PNG, allowing transparency.
# - Overlay the arrow PNG to the lane image overlay.


"""
Title: Alternate simpler method for directly extracting points
Author: GeeksForGeeks
Date: 07/25/2024
Type: Source Code
Availability: https://www.geeksforgeeks.org/line-detection-python-opencv-houghline-method/
"""

"""
Title: Live Webcam Drawing using OpenCV
Author: GeeksForGeeks
Date: 01/03/2023
Type: Source Code
Availability: https://www.geeksforgeeks.org/live-webcam-drawing-using-opencv/
"""

"""
Title: Image Thresholding
Author: OpenCV
Type: Source Code
Availability: https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
"""

"""
Title: Contour Features
Author: OpenCV
Type: Source Code
Availability: https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
"""

"""
Title: Overlay a Smaller Image on a Larger Image Python OpenCV
Author: fireant, Mateen Ulhaq
Date: 12/31/2012
Type: Source Code
Availability: https://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencv
"""


import cv2
import numpy as np


frame_count = 0
neg_lanes = []
pos_lanes = []

arrow_degree = "UP"


def detect_shapes(img, arrow_img, height, width):
    """Returns an image with an overlay of lane lines, their centerline, and a compass

    Detects lane lines in the input image, draws two blue parallel lines and a yellow centerline, detects arrows on the
    road, rotates the compass based on the turn indicated by the arrow, and returns the image overlay

    Parameters
    -----------
        img: numpy.ndarray
            An individual frame of the road video
        arrow_img: numpy.ndarray
            A PNG of a red arrow used as a compass
        height: int
            The height of the video
        width: int
            The width of the video
    Return
    -----------
        arrow_img: numpy.ndarray
            The final image with all the image overlay
    """

    global arrow_degree

    line_img = preprocess(img, height, width)  # Preprocess the raw image for detection
    contour_img = preprocess(img, height, width)  # Preprocess the raw image for detection

    try:
        line_img = cv2.Canny(line_img, 100, 200, apertureSize=7)

        points = np.array([[int(width * 0.25), int(height * 0.65)], [int(width * 0.05), int(height * 0.90)],
                           [int(width * 0.30), int(height * 0.90)], [int(width * 0.40), int(height * 0.65)],
                           [int(width * 0.53), int(height * 0.90)], [int(width * 0.82), int(height * 0.90)],
                           [int(width * 0.55), int(height * 0.65)]])

        trap_mask = np.zeros(line_img.shape[:2], dtype="uint8")
        cv2.fillPoly(trap_mask, [points], (255, 255, 255))
        line_img = cv2.bitwise_and(line_img, line_img, mask=trap_mask)

        binr = cv2.threshold(line_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        kernel = np.ones((3, 3), np.uint8)
        invert = cv2.bitwise_not(binr)

        morph_gradient = cv2.morphologyEx(invert, cv2.MORPH_GRADIENT, kernel, iterations=1)

        kernel = np.ones((3, 3), np.uint8)
        morph_gradient = cv2.erode(morph_gradient, kernel, iterations=1)
        kernel = np.ones((5, 5), np.uint8)
        morph_gradient = cv2.dilate(morph_gradient, kernel, iterations=1)

        lines = cv2.HoughLinesP(morph_gradient, 1, np.pi / 180, threshold=50, minLineLength=20, maxLineGap=30)
        pos_line = []
        neg_line = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            if x1 != x2:
                m = (y2 - y1) / (x2 - x1)
                if 0.7 < abs(m) < 3:
                    b = y1 - (m * x1)
                    if m < 0 and x1 < width * 0.4 and x2 < width * 0.4:
                        neg_line.append([m, b])
                    elif m > 0 and x1 > width * 0.4 and x2 > width * 0.4:
                        pos_line.append([m, b])

        global frame_count

        if len(neg_line) > 0:
            neg_line.sort(key=lambda neg: neg[0])
            neg_avg_m = neg_line[len(neg_line) // 2][0]
            neg_line.sort(key=lambda neg: neg[1])
            neg_avg_b = neg_line[len(neg_line) // 2][1]

            if frame_count != 0 and ((neg_lanes[frame_count - 1][0] * 0.7) <= neg_avg_m or
                                     (neg_lanes[frame_count - 1][0] * 1.3) >= neg_avg_m):
                neg_avg_m = neg_lanes[frame_count - 1][0]
                neg_avg_b = neg_lanes[frame_count - 1][1]
        else:
            neg_avg_m = neg_lanes[frame_count - 1][0]
            neg_avg_b = neg_lanes[frame_count - 1][1]

        if len(pos_line) > 0:
            pos_line.sort(key=lambda pos: pos[0])
            pos_avg_m = pos_line[len(pos_line) // 2][0]
            pos_line.sort(key=lambda pos: pos[1])
            pos_avg_b = pos_line[len(pos_line) // 2][1]

            if frame_count != 0 and ((pos_lanes[frame_count - 1][0] * 0.7) >= pos_avg_m or
                                     (pos_lanes[frame_count - 1][0] * 1.3) <= pos_avg_m):
                pos_avg_m = pos_lanes[frame_count - 1][0]
                pos_avg_b = pos_lanes[frame_count - 1][1]
        else:
            pos_avg_m = pos_lanes[frame_count - 1][0]
            pos_avg_b = pos_lanes[frame_count - 1][1]

        cv2.line(img, (int((height * 0.6 - neg_avg_b) / neg_avg_m), int(height * 0.6)),
                 (int((height - neg_avg_b) / neg_avg_m), height), (255, 0, 0), 2)
        cv2.line(img, (int((height * 0.6 - pos_avg_b) / pos_avg_m), int(height * 0.6)),
                 (int((height - pos_avg_b) / pos_avg_m), height),
                 (255, 0, 0), 2)
        cv2.line(img, (
            int((((height * 0.6 - neg_avg_b) / neg_avg_m) + ((height * 0.6 - pos_avg_b) / pos_avg_m)) / 2),
            int(height * 0.6)),
                 (int((((height - neg_avg_b) / neg_avg_m) + ((height - pos_avg_b) / pos_avg_m)) / 2), height),
                 (0, 255, 255), 2)

        neg_lanes.append([neg_avg_m, neg_avg_b])
        pos_lanes.append([pos_avg_m, pos_avg_b])

        frame_count += 1
    except:
        pass

    try:
        contour_img = cv2.Canny(contour_img, 50, 200, apertureSize=7)

        points = np.array([[int(width * 0.25), int(height * 0.75)], [int(width * 0.15), int(height * 0.90)],
                           [int(width * 0.70), int(height * 0.90)], [int(width * 0.57), int(height * 0.75)]], np.int32)

        trap_mask = np.zeros(contour_img.shape[:2], dtype="uint8")
        cv2.fillPoly(trap_mask, [points], (255, 255, 255))
        contour_img = cv2.bitwise_and(contour_img, contour_img, mask=trap_mask)

        kernel = np.ones((5, 5), np.uint8)
        contour_img = cv2.dilate(contour_img, kernel, iterations=1)
        kernel = np.ones((3, 3), np.uint8)
        contour_img = cv2.erode(contour_img, kernel, iterations=1)

        binr = cv2.threshold(contour_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        kernel = np.ones((3, 3), np.uint8)
        invert = cv2.bitwise_not(binr)

        morph_gradient = cv2.morphologyEx(invert, cv2.MORPH_GRADIENT, kernel, iterations=1)

        kernel = np.ones((3, 3), np.uint8)
        morph_gradient = cv2.dilate(morph_gradient, kernel, iterations=1)
        morph_gradient = cv2.erode(morph_gradient, kernel, iterations=1)

        contours, hierarchy = cv2.findContours(morph_gradient, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            epsilon = 0.00001 * cv2.arcLength(contour, True)
            approximate_contour = cv2.approxPolyDP(contour, epsilon, True)

            if (cv2.arcLength(approximate_contour, True) < 200 or cv2.arcLength(approximate_contour, True) > 1000) or (
                    cv2.contourArea(approximate_contour) < 6000):
                continue

            rect = cv2.minAreaRect(approximate_contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            if 15000 <= cv2.contourArea(box) < 20000:
                if 10 <= rect[2] <= 20:
                    cv2.drawContours(img, [box], 0, (255, 0, 255), 2)  # Magenta = Left
                    arrow_degree = "LEFT"
                elif 70 <= rect[2] <= 80:
                    cv2.drawContours(img, [box], 0, (0, 255, 255), 2)  # Yellow = Right
                    arrow_degree = "RIGHT"
                else:
                    continue
            else:
                if 10000 <= cv2.contourArea(box) < 15000 or 22000 <= cv2.contourArea(box) < 23000:
                    arrow_degree = "UP"
                continue

            cv2.drawContours(img, [approximate_contour], -1, (0, 0, 255), 2)


    except:
        pass

    arrow = cv2.resize(arrow_img, (0, 0), fx=0.5, fy=0.5)
    arrow = rotate_arrow(arrow, arrow_degree)
    h, w = arrow.shape[:2]

    arrow_img = img.copy()
    arrow_img = cv2.cvtColor(arrow_img, cv2.COLOR_BGR2BGRA)

    arrow_alpha = arrow[:, :, 3] / 255.0
    arrow_img_alpha = 1.0 - arrow_alpha

    for channel in range(0, 3):
        arrow_img[:h, width - w:, channel] = (
                arrow_alpha * arrow[:, :, channel] + arrow_img_alpha * arrow_img[:h, width - w:, channel])

    return arrow_img


def preprocess(img, height, width):
    """Returns an image suited for line detection

    Performs various operations on an image to make it better for lane detection

    Parameters
    -----------
        img: numpy.ndarray
            The given image for formatting
        height: int
            The height of the image
        width: int
            The width of the image
    Return
    -----------
        gray_img: numpy.ndarray
            The formatted image, ready for detection
    """
    gray_img = img.copy()
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
    blue_img = np.zeros((height, width, 3), dtype="uint8")
    blue_img[:, :, 0] = gray_img

    lower_blue = np.array([130, 0, 0], dtype="uint8")
    upper_blue = np.array([170, 0, 0], dtype="uint8")
    blue_mask = cv2.inRange(blue_img, lower_blue, upper_blue)
    blue_img = cv2.bitwise_and(blue_img, blue_img, mask=blue_mask)

    gray_img = cv2.cvtColor(blue_img, cv2.COLOR_RGB2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    kernel = np.ones((11, 11), np.uint8)
    gray_img = cv2.dilate(gray_img, kernel, iterations=1)
    kernel = np.ones((13, 13), np.uint8)
    gray_img = cv2.erode(gray_img, kernel, iterations=1)

    thresh_mask = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, -1)
    gray_img = cv2.bitwise_and(gray_img, gray_img, mask=thresh_mask)
    kernel = np.ones((9, 9), np.uint8)
    gray_img = cv2.dilate(gray_img, kernel, iterations=1)
    kernel = np.ones((5, 5), np.uint8)
    gray_img = cv2.erode(gray_img, kernel, iterations=2)

    return gray_img


def rotate_arrow(arrow, deg):
    """Returns an arrow that is rotated the specified amount

    Rotates an arrow and returns the product

    Parameters
    -----------
        arrow: numpy.ndarray
            The arrow PNG that will be rotated
        deg: str
            The direction of the arrow after rotation
    Return
    -----------
        cv2.rotate(arrow, cv2.ROTATE_90_COUNTERCLOCKWISE): numpy.ndarray
            An arrow pointing up
        arrow: numpy.ndarray
            An arrow pointing to the right
        cv2.rotate(arrow, cv2.ROTATE_90_CLOCKWISE): numpy.ndarray
            An arrow pointing down
        cv2.rotate(arrow, cv2.ROTATE_180): numpy.ndarray
            An arrow pointing to the left

    """
    if deg == "UP":
        return cv2.rotate(arrow, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif deg == "RIGHT":
        return arrow
    elif deg == "DOWN":
        return cv2.rotate(arrow, cv2.ROTATE_90_CLOCKWISE)
    elif deg == "LEFT":
        return cv2.rotate(arrow, cv2.ROTATE_180)
