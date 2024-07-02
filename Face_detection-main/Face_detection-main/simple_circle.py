import cv2

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
        # Draw a circle around each detected face
        center_face = (x + w//2, y + h//2)
        radius_face = int(min(w, h) // 2)
        cv2.circle(img, center_face, radius_face, (255, 255, 0), 2)
        
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        # Detect eyes within each detected face
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            # Draw a circle around each detected eye
            center_eye = (x + ex + ew//2, y + ey + eh//2)
            radius_eye = int(min(ew, eh) // 2)
            cv2.circle(img, center_eye, radius_eye, (0, 127, 255), 2)

    # Display the result
    cv2.imshow('img', img)

    # Exit the loop if the 'Esc' key is pressed
    if cv2.waitKey(30) & 0xff == 27:
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
