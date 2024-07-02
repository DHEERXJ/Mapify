import cv2

# Load the Haar cascades for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Verify that the cascade has loaded successfully
if face_cascade.empty():
    print("Error loading face cascade")

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
        # Extract the region of interest (ROI) for the face
        face_roi = img[y:y + h, x:x + w]
        
        # Apply a blur to the face ROI
        face_roi_blurred = cv2.GaussianBlur(face_roi, (99, 99), 30)
        
        # Replace the face region in the image with the blurred face
        img[y:y + h, x:x + w] = face_roi_blurred

    # Display the result
    cv2.imshow('img', img)

    # Exit the loop if the 'Esc' key is pressed
    if cv2.waitKey(30) & 0xff == 27:
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
