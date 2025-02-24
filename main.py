# Aldrei Chua 02/10/2025
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
# import matplotlib.pyplot as plt
import numpy as np

# Get the video feed from the web camera
video = cv2.VideoCapture("AldreiC - RoadVideo.MOV")


def detect_shapes(img):
    # Detect lines in the input image and output the image with two blue parallel lines and a red centerline

    # Preprocess the raw image for detection
    gray_img = preprocess(img)

    try:
        lines = cv2.HoughLinesP(gray_img, 1, np.pi / 180, threshold=10, minLineLength=25, maxLineGap=20)
        pos_line = [0, 0, 0]
        neg_line = [0, 0, 0]

        for line in lines:
            x1, y1, x2, y2 = line[0]

            if x1 != x2:
                m = (y2 - y1) / (x2 - x1)
                if 0.4 < abs(m) < 5:
                    b = y1 - (m * x1)
                    if m < 0:
                        neg_line[0] += m
                        neg_line[1] += b
                        neg_line[2] += 1
                    elif m > 0:
                        pos_line[0] += m
                        pos_line[1] += b
                        pos_line[2] += 1

        if neg_line[2] != 0 and pos_line[2] != 0:
            neg_avg_m = neg_line[0] / neg_line[2]
            neg_avg_b = neg_line[1] / neg_line[2]
            pos_avg_m = pos_line[0] / pos_line[2]
            pos_avg_b = pos_line[1] / pos_line[2]

            cv2.line(img, (int((height * 0.6 - neg_avg_b) / neg_avg_m), int(height * 0.6)),
                     (int((height - neg_avg_b) // neg_avg_m), height), (255, 0, 0), 2)
            cv2.line(img, (int((height * 0.6 - pos_avg_b) / pos_avg_m), int(height * 0.6)),
                     (int((height - pos_avg_b) / pos_avg_m), height),
                     (255, 0, 0), 2)
            cv2.line(img, (
                int((((height * 0.6 - neg_avg_b) / neg_avg_m) + ((height * 0.6 - pos_avg_b) / pos_avg_m)) / 2),
                int(height * 0.6)),
                     (int((((height - neg_avg_b) / neg_avg_m) + ((height - pos_avg_b) / pos_avg_m)) / 2), height),
                     (0, 255, 255), 2)


    except:
        pass

    try:
        circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, 100, param1=150, param2=40, minRadius=50,
                                   maxRadius=200)
        try:
            circles = np.uint16(np.around(circles))
        except:
            circles = []

        # For each detected circle in the array, overlay a blue circle with the same dimensions along with a red center
        try:
            for circle in circles[0][:]:
                cv2.circle(img, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
                cv2.circle(img, (circle[0], circle[1]), 1, (255, 0, 255), 3)


        except:
            pass
    except:
        pass

    try:
        contours, hierarchy = cv2.findContours(gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_list = []

        # Detect all long, curved contours around the middle of the screen
        for contour in contours:
            epsilon = 0.001 * cv2.arcLength(contour, False)
            approximate_contour = cv2.approxPolyDP(contour, epsilon, False)

            length = cv2.arcLength(approximate_contour, False)
            coordinate_total = approximate_contour[0][0][0] + approximate_contour[0][0][1]
            if length <= 500 or coordinate_total <= (width + height) * 0.2 or coordinate_total >= (
                    width + height) * 0.8:
                continue

            contour_list.append(approximate_contour)
            cv2.drawContours(img, [approximate_contour], -1, (0, 0, 255), 2)

        # Find and display the midline between two contours
        c_line1 = contour_list[0]
        c_line2 = contour_list[1]

        midline = []
        for point1, point2 in zip(c_line1, c_line2):
            m_pointx = (point1[0][0] + point2[0][0]) // 2
            m_pointy = (point1[0][1] + point2[0][1]) // 2
            midline.append([m_pointx, m_pointy])

        midline = np.array(midline, dtype=np.int32)
        cv2.polylines(img, [midline], isClosed=False, color=(255, 255, 0), thickness=2)
    except:
        pass

    return img


def preprocess(img):
    gray_img = img.copy()
    hsv = cv2.cvtColor(gray_img, cv2.COLOR_BGR2HSV)
    hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    lower_white = np.array([0, 100, 120], dtype="uint8")
    upper_white = np.array([255, 160, 160], dtype="uint8")
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    lower_gray = np.array([0, 170, 100], dtype="uint8")
    upper_gray = np.array([255, 200, 220], dtype="uint8")
    gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
    lower_yellow = np.array([20, 100, 50], dtype="uint8")
    upper_yellow = np.array([80, 255, 255], dtype="uint8")
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    combined_mask = cv2.bitwise_or(white_mask, gray_mask)
    combined_mask = cv2.bitwise_or(combined_mask, yellow_mask)
    hsv = cv2.bitwise_and(hsv, hsv, mask=combined_mask)
    cv2.imshow("hsv_after", hsv)

    kernel = np.ones((7, 7), np.uint8)
    hsv = cv2.erode(hsv, kernel, iterations=1)

    gray_img = cv2.cvtColor(hsv, cv2.COLOR_RGB2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (7, 7), 0)

    kernel = np.ones((11, 11), np.uint8)
    gray_img = cv2.dilate(gray_img, kernel, iterations=1)
    gray_img = cv2.erode(gray_img, kernel, iterations=1)

    thresh_mask = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, -1)
    gray_img = cv2.bitwise_and(gray_img, gray_img, mask=thresh_mask)

    kernel = np.ones((5, 5), np.uint8)
    gray_img = cv2.Canny(gray_img, 50, 150, L2gradient=False)
    gray_img = cv2.dilate(gray_img, kernel, iterations=1)
    gray_img = cv2.erode(gray_img, kernel, iterations=1)

    points = np.array([[int(width * 0.33), int(height * 0.65)], [int(width * 0.10), int(height * 0.88)],
                       [int(width * 0.77), int(height * 0.88)], [int(width * 0.55), int(height * 0.65)]])

    trap_mask = np.zeros(gray_img.shape[:2], dtype="uint8")
    cv2.fillPoly(trap_mask, [points], (255, 255, 255))
    gray_img = cv2.bitwise_and(gray_img, gray_img, mask=trap_mask)

    return gray_img


def show_arrow(img):
    arrow = cv2.imread("Arrow.png", -1)
    h, w = arrow.shape[:2]

    arrow_img = img.copy()
    arrow_img = cv2.cvtColor(arrow_img, cv2.COLOR_BGR2BGRA)

    arrow_alpha = arrow[:, :, 3] / 255.0 + 0.01
    arrow_img_alpha = 1.0 - arrow_alpha

    for channel in range(0, 3):
        arrow_img[:h, width - w:, channel] = (
                    arrow_alpha * arrow[:, :, channel] + arrow_img_alpha * arrow_img[:h, width - w:, channel])

    cv2.imshow('Lane Detection', arrow_img)


while True:
    # Constantly reads the webcam feed and creates individual images for processing

    _, image = video.read()
    height, width = image.shape[:2]
    imageResult = detect_shapes(image)
    show_arrow(image)

    # Closes the window when the space bar is pressed
    if cv2.waitKey(1) == ord(' '):
        break
