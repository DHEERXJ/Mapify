'''
Facial Landmark Detection in Python with OpenCV

Detection from webcam
'''

# Import Packages
import cv2
import os
import urllib.request as urlreq
import numpy as np

# Haarcascade for face detection
haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
haarcascade = "haarcascade_frontalface_alt2.xml"
haarcascade_clf = "data/" + haarcascade

# Ensure the Haarcascade is downloaded
if not os.path.isdir('data'):
    os.mkdir('data')

if haarcascade not in os.listdir('data'):
    urlreq.urlretrieve(haarcascade_url, haarcascade_clf)

# Create an instance of the Face Detection Cascade Classifier
detector = cv2.CascadeClassifier(haarcascade_clf)

# LBF model for facial landmark detection
LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"
LBFmodel = "lbfmodel.yaml"
LBFmodel_file = "data/" + LBFmodel

# Ensure the LBF model is downloaded
if LBFmodel not in os.listdir('data'):
    urlreq.urlretrieve(LBFmodel_url, LBFmodel_file)

# Create an instance of the Facial landmark Detector with the model
landmark_detector = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel_file)

# Check webcam connection
print("Checking webcam for connection ...")
webcam_cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = webcam_cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using the Haarcascade classifier on the grayscale image
    faces = detector.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        # Detect landmarks
        _, landmarks = landmark_detector.fit(gray, faces)

        for landmark in landmarks:
            # Draw circles at each landmark point
            for x, y in landmark[0]:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)  # Circle

            # Draw lines between the landmarks
            for i in range(0, 68):
                if i in [16, 21, 26, 30, 35, 41, 47, 59, 67]:  # End points of certain regions
                    continue
                x1, y1 = landmark[0][i]
                x2, y2 = landmark[0][i + 1]
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)

            # Draw lines for specific regions (connect the first to the last point of each region)
            regions = {
                'jaw': range(0, 17),
                'right_eyebrow': range(17, 22),
                'left_eyebrow': range(22, 27),
                'nose_bridge': range(27, 31),
                'lower_nose': range(31, 36),
                'right_eye': range(36, 42),
                'left_eye': range(42, 48),
                'outer_lip': range(48, 60),
                'inner_lip': range(60, 68)
            }

            for region in regions.values():
                points = landmark[0][list(region)]
                for j in range(len(points) - 1):
                    cv2.line(frame, (int(points[j][0]), int(points[j][1])), (int(points[j + 1][0]), int(points[j + 1][1])), (255, 0, 0), 1)
                # Close the loop by connecting the last point to the first
                cv2.line(frame, (int(points[-1][0]), int(points[-1][1])), (int(points[0][0]), int(points[0][1])), (255, 0, 0), 1)

    # Save the last instance of the detected image
    cv2.imwrite('face-detect.jpg', frame)

    # Show image
    cv2.imshow("frame", frame)

    # Terminate the capture window
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
webcam_cap.release()
cv2.destroyAllWindows()
