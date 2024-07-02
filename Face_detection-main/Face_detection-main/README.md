# Facial Landmark Detection with OpenCV

This project demonstrates how to detect facial landmarks in real-time using OpenCV. It uses a pre-trained Haarcascade classifier for face detection and a pre-trained LBF (Local Binary Features) model for facial landmark detection. The application captures frames from a webcam, detects faces and their landmarks, and displays them with connected lines and circles.

## Features

- Real-time face detection using Haarcascade classifier.
- Real-time facial landmark detection using the LBF model.
- Visualization of facial landmarks with both circles (dots) and lines connecting the points.
- Saves the last captured frame with detected landmarks.

## Requirements

- Python 3.7+
- OpenCV (cv2)
- NumPy
- Internet connection to download the necessary models

## Setup Instructions

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/yourusername/facial-landmark-detection.git
    cd facial-landmark-detection
    ```

2. **Install the Required Packages:**

    You can install the required Python packages using pip:

    ```bash
    pip install opencv-python opencv-contrib-python numpy
    ```

3. **Download the Pre-trained Models:**

    The script automatically downloads the Haarcascade classifier for face detection and the LBF model for facial landmark detection if they are not already present in the `data` directory.

## Usage

1. **Run the Script:**

    ```bash
    python facial_landmark_detection.py
    ```

2. **Functionality:**

    - The script opens a window showing the live feed from your webcam.
    - Detected faces and their landmarks are displayed with circles and lines.
    - Press `q` to quit the application and close the window.

3. **Saved Output:**

    - The script saves the last captured frame with detected landmarks as `face-detect.jpg` in the current directory.

## Code Overview

```python
'''
Facial Landmark Detection in Python with OpenCV

Detection from webcam
'''

# Import Packages
import cv2
import os
import urllib.request as urlreq
import numpy as np
