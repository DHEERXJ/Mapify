import cv2
import numpy as np

# Load the Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Verify that the cascades have loaded successfully
if face_cascade.empty():
    print("Error loading face cascade")
if eye_cascade.empty():
    print("Error loading eye cascade")

# Capture video from the default camera (index 0)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, img = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Create an overlay image
        overlay = img.copy()
        
        # Draw a filled ellipse to cover the face area
        center = (x + w//2, y + h//2)
        axes = (w//2, h//2)
        cv2.ellipse(overlay, center, axes, 0, 0, 360, (0, 0, 0), -1)  # Black color mask
        
        # Blend the overlay with the original image with a transparency factor
        alpha = 0.4  # Transparency factor
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Display the result
    cv2.imshow('img', img)

    # Exit the loop if the 'Esc' key is pressed
    if cv2.waitKey(30) & 0xff == 27:
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
