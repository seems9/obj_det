import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

# Load the EfficientDet model from TensorFlow Hub
model = hub.load("https://tfhub.dev/tensorflow/efficientdet/d1/1")


image_path = "C:\\Users\\seems\\test_img.jpg"  # Change this to your image path
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
# print("Img:", image)
# Resize image while keeping dtype uint8
image_resized = cv2.resize(image, (512, 512)).astype(np.uint8)

input_tensor = tf.convert_to_tensor(image_resized, dtype=tf.uint8)  # Ensure uint8
input_tensor = tf.expand_dims(input_tensor, axis=0)  # Model expects (1, H, W, 3)

# Run inference
detections = model.signatures["serving_default"](input_tensor) 

boxes = detections['detection_boxes'][0].numpy()  # (100, 4)
classes = detections['detection_classes'][0].numpy().astype(int)  # (100,)
scores = detections['detection_scores'][0].numpy()  # (100,)

# Define threshold for detection confidence
confidence_threshold = 0.5

# Convert normalized box coordinates to pixel values
height, width, _ = image.shape
boxes = boxes * [height, width, height, width]  # Scale to image size

# Filter out low-confidence detections
valid_detections = [(box, cls, score) for box, cls, score in zip(boxes, classes, scores) if score > confidence_threshold]

# Print filtered results
for box, cls, score in valid_detections:
    print(f"Class: {cls}, Confidence: {score:.2f}, Box: {box}")

# Class labels (COCO dataset used by EfficientDet)
CLASS_NAMES = {1: "Person", 2: "Bicycle", 3: "Car", 4: "Motorcycle", 5: "Airplane"}  # Add more as needed

for box, cls, score in valid_detections:
    ymin, xmin, ymax, xmax = box.astype(int)
    
    # Draw bounding box
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    
    # Put class label & confidence
    label = f"{CLASS_NAMES.get(cls, 'Unknown')} ({score:.2f})"
    cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show image
cv2.imshow("Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
