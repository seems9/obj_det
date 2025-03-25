# Object Detection using EfficientDet and TensorFlow Hub

Overview

This project implements an object detection system using TensorFlow Hub's EfficientDet model. It loads an image, performs inference, and visualizes the detected objects with bounding boxes and labels.

Features

Uses EfficientDet D1 model from TensorFlow Hub for object detection.

Processes input images using OpenCV.

Filters detected objects based on a confidence threshold.

Draws bounding boxes and labels on detected objects.

Displays the processed image with detected objects.

Dependencies

Ensure you have the following Python packages installed:

pip install tensorflow tensorflow_hub numpy opencv-python

Setup and Usage

Install Dependencies: Run the above pip install command to install required libraries.

Modify the Image Path: Update image_path in objdet.py to point to your input image.

Run the Script: Execute the following command:

python objdet.py

View Results: The script will display the image with detected objects.

How It Works

Load Model: EfficientDet model is loaded from TensorFlow Hub.

Preprocess Image:

Read the input image.

Convert it to RGB format.

Resize it to match the model's expected input size.

Run Inference:

Convert the image to a tensor.

Pass it through the model.

Extract bounding boxes, class labels, and confidence scores.

Filter Results: Only detections with a confidence score above 0.5 are kept.

Draw Bounding Boxes:

Convert normalized box coordinates to pixel values.

Draw rectangles around detected objects.

Label them with their class names and confidence scores.

Display Output: The final processed image is displayed using OpenCV.

Code Breakdown

model = hub.load(): Loads the EfficientDet model from TensorFlow Hub.

cv2.imread(): Reads the input image.

cv2.resize(): Resizes the image to the required model input size.

model.signatures["serving_default"](input_tensor): Runs inference.

cv2.rectangle(): Draws bounding boxes around detected objects.

cv2.putText(): Displays object class labels and confidence scores.

cv2.imshow(): Displays the processed image.

Future Enhancements

Support for real-time object detection using a webcam.

Extend the model to detect additional object classes.

Save the processed image with bounding boxes and labels.

Optimize inference time for better performance.
