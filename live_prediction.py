# Standard library imports
import os
from datetime import datetime, timedelta
import threading
import queue
import pickle
from joblib import load

# Third-party library imports
import cv2
from ultralytics import YOLO
from dotenv import load_dotenv

# Local application/library-specific imports
from email_notifier import send_email_notification


# Load environment variables
load_dotenv()
rtsp_url = os.getenv("outdoor_camera")

# Initialize YOLO model, Load Logistic Regression model and PCA model
model = YOLO("yolo11n.pt")

# Loading the logistic Regression Model
logistic_model = load("Models/logistic_regression_model.joblib")
print("Logistic Regression Model loaded successfully")

# Loading the PCA Model
pca_model = load("Models/pca_model.joblib")
print("PCA Model loaded successfully")

"""
with open('logistic_regression_model.pkl', 'rb') as lr_file:
    logistic_model = pickle.load(lr_file)

with open('pca_model.pkl', 'rb') as pca_file:
    pca_model = pickle.load(pca_file)
"""

# Create a queue for frames
frame_queue = queue.Queue(maxsize=10)

# Shared variable to track last email time
last_email_time = datetime.min
email_lock = threading.Lock()  # Ensures thread-safe access to `last_email_time`

# Frame capture thread
def capture_frames():
    cap = cv2.VideoCapture("Videos\F48318325_1_20250116T194305Z_20250116T194505Z.mp4")
    if not cap.isOpened():
        print("Failed to connect to camera.")
        return 
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to retrieve frame.")
            break
        if not frame_queue.full():
            frame_queue.put(frame)
    cap.release()

# Email sending function
def send_email_if_needed(frame, score):
    global last_email_time
    with email_lock:  # Ensure thread-safe access to `last_email_time`
        current_time = datetime.now()
        if current_time - last_email_time >= timedelta(minutes=5):
            # Save the frame as an image
            timestamp = current_time.strftime("%Y%m%d_%H%M%S")
            image_path = f"bully_cat_images/{timestamp}.jpg"
            os.makedirs("bully_cat_images", exist_ok=True)
            cv2.imwrite(image_path, frame)

            # Send the email
            send_email_notification(image_path, score)
            last_email_time = current_time
            print(f"Email sent at {current_time}.")
        else:
            print("Email not sent. Cooldown period active.")

# Frame processing thread
def process_frames():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            # Run YOLO model on the frame
            results = model.predict(frame, verbose=False)

            # Process detections for all objects
            for detection in results[0].boxes:

                if detection.cls[0] == 15:
                    box = detection.xyxy[0].tolist()  # Bounding box (x_min, y_min, x_max, y_max)
                    score = detection.conf[0]  # Confidence score
                    class_id = int(detection.cls[0])  # Class ID
                    label = f"{model.names[class_id]} {score:.2f}"  # Class label with confidence

                    if "cat" in label:
                        x_min, y_min, x_max, y_max = map(int, box)

                        # Draw the bounding box and label on the frame
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                        # Resizing and flattening the image
                        cropped_frame = frame[y_min:y_max, x_min:x_max]
                        resized_image = cv2.resize(cropped_frame, (255, 255))
                        flattened_image = resized_image.flatten()

                        # Apply PCA
                        reduced_features = pca_model.transform([flattened_image])

                        # Logistic Regression model prediction
                        prediction = logistic_model.predict(reduced_features)
                        confidence = logistic_model.predict_proba(reduced_features).max()

                        # Display the prediction and confidence
                        print(f"Predicted Class: {prediction[0]}, Confidence: {confidence:.2f}")

                        # Check if email needs to be sent
                        send_email_if_needed(frame, confidence)

            # Display the annotated frame
            cv2.imshow("Object Detection", frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()

# Start threads
capture_thread = threading.Thread(target=capture_frames)
process_thread = threading.Thread(target=process_frames)
capture_thread.start()
process_thread.start()
capture_thread.join()
process_thread.join()