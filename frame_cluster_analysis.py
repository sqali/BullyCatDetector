# Standard library imports
import os
from datetime import datetime

# Third-party library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# Scikit-learn imports
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump

# Other third-party imports
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO("yolo11n.pt")

def store_pixel_info_with_labels():
    # Define the directory paths for each label
    frame_directory = ["frame_directory/grey", "frame_directory/orange"]
    
    for directory_number, directory in enumerate(frame_directory):
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            continue

        files = os.listdir(directory)

        if not files:
            print(f"No files found in the directory: {directory}")
            continue

        for file_number, file in enumerate(files):
            file_path = os.path.join(directory, file)

            if not os.path.isfile(file_path):
                print(f"Invalid file: {file_path}")
                continue

            print(f"Processing file {file_number}: {file_path}")

            image = cv2.imread(file_path)

            if image is None:
                print(f"Failed to read image: {file_path}")
                continue
            else:
                # Resize the image to the target size (255, 255)
                resized_image = cv2.resize(image, (255, 255))
                
                # Flatten the image into a 1D array
                flattened_image = resized_image.flatten()
                
                # Assign the label based on the directory (grey or orange)
                if "grey" in directory:
                    label = "Grey"
                else:
                    label = "Orange"
                
                # Save the flattened image as a numpy array
                np.save(f"pixel_arrays/{label}/{file[:-4]}.npy", flattened_image)

def create_pca():

    # Define the directory paths for each label
    grey_directory = "pixel_arrays/Grey"
    orange_directory = "pixel_arrays/Orange"
    
    # Initialize lists to hold the data and labels
    data = []
    labels = []
    
    # Function to process each directory and collect the data
    def process_directory(directory, label):
        for file in os.listdir(directory):
            if file.endswith(".npy"):
                # Load the image data
                file_path = os.path.join(directory, file)
                image = np.load(file_path)
                data.append(image.flatten())  # Flatten the image to a 1D array
                labels.append(label)
    
    # Process the Grey and Orange directories
    process_directory(grey_directory, "Grey")
    process_directory(orange_directory, "Orange")
    
    # Convert data to numpy array
    data = np.array(data)
    labels = np.array(labels)
    
    # Apply PCA to reduce the dimensions
    pca = PCA(n_components=2)  # Reduce to 2 components for easy visualization
    pca_data = pca.fit_transform(data)  # Apply PCA to the data
    
    # Create a DataFrame with PCA features and labels
    pca_df = pd.DataFrame(pca_data, columns=["PCA1", "PCA2"])
    pca_df['Label'] = labels
    
    # Save the resulting DataFrame to a CSV file
    pca_df.to_csv("Dataset/pca_feature_dataset_2d_augmentation.csv", index=False)
    print("PCA feature dataset saved as 'pca_feature_dataset_2d_augmentation.csv'.")

    # Save the PCA model for later use
    dump(pca, 'Models/pca_model.joblib')  # Save the PCA model as a .joblib file
    print("PCA model saved as 'pca_model.joblib'.")


def visualize_pca():
    """
    Loads the PCA-transformed dataset from a CSV file and visualizes it using a scatter plot.

    Parameters:
        None
    """
    # Load the PCA dataset
    pca_df = pd.read_csv("Dataset/pca_feature_final_dataset.csv")

    print(pca_df.columns)

    # Check if the dataset contains at least two PCA components
    if 'PCA1' not in pca_df.columns or 'PCA2' not in pca_df.columns:
        raise ValueError("The provided dataset does not contain the required PCA components 'PCA1' and 'PCA2'.")

    # Plotting the PCA components with hue based on 'Label' - Two Feature Based PCA
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Label', data=pca_df, palette=['grey', 'orange'], style='Label', s=100, markers=['o', 's'])
    plt.title('PCA Visualization of Cat Images (Orange vs Grey)', fontsize=16)
    plt.show()


def logistic_regression_training():
    # Load PCA dataset
    pca_df = pd.read_csv("Dataset/pca_feature_final_dataset.csv")
    
    # Separate features and labels
    X = pca_df[['PCA1', 'PCA2']].values
    y = pca_df['Label'].values
    
    # Encode labels if they are categorical
    if y.dtype == 'O':  # Check if labels are strings
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    
    # Plot training points
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', edgecolor='k', s=80, label='Training Data')
    
    # Logistic regression coefficients
    beta_0 = model.intercept_[0]
    beta_1, beta_2 = model.coef_[0]
    
    # Define x1 range based on data range with margin
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x1_margin = (x1_max - x1_min) * 0.2
    x1_range = np.linspace(x1_min - x1_margin, x1_max + x1_margin, 500)
    
    # Calculate x2 (decision boundary) for the logistic regression equation
    x2_boundary = -(beta_0 + beta_1 * x1_range) / beta_2
    
    # Restrict to valid data ranges for plotting
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    valid_mask = (x2_boundary >= x2_min) & (x2_boundary <= x2_max)
    x1_range = x1_range[valid_mask]
    x2_boundary = x2_boundary[valid_mask]

    # Saving the Logistic Regression model for live prediction
    dump(model, 'Models/logistic_regression_model.joblib')  # Save the Logistic Regression model as a .joblib file
    
    # Plot the decision boundary
    plt.plot(x1_range, x2_boundary, color='red', label='Decision Boundary')
    
    # Add labels and legend
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.title('Logistic Regression Decision Boundary and Training Data')
    plt.legend()
    plt.show()

# Frame capture thread
def frame_pipeline():
    video_directory = ["Videos/Grey", "Videos/Orange"]

    count = 0  # For directory switching
    for directory in video_directory:
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            continue

        files = os.listdir(directory)
        if not files:
            print(f"No files found in directory: {directory}")
            continue

        for i, file in enumerate(files):
            file_path = os.path.join(directory, file)
            if not os.path.isfile(file_path):
                print(f"Invalid file: {file_path}")
                continue

            print(f"Processing file {i}: {file_path}")

            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                print(f"Failed to open video file: {file_path}")
                continue

            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"End of video or failed to retrieve frame: {file_path}")
                    break

                # Run YOLO model on the frame
                results = model.predict(frame, verbose=False)

                # Process detections for all objects
                for detection in results[0].boxes:
                    if detection.cls[0] == 15:  # Example class ID
                        box = detection.xyxy[0].tolist()  # Bounding box (x_min, y_min, x_max, y_max)
                        score = detection.conf[0]  # Confidence score
                        class_id = int(detection.cls[0])  # Class ID
                        label = f"{model.names[class_id]} {score:.2f}"  # Class label with confidence

                        if "cat" in label:
                            x_min, y_min, x_max, y_max = map(int, box)
                            
                            # Crop the region of interest and save that frame
                            roi = frame[y_min:y_max, x_min:x_max]

                            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                            save_dir = "frame_directory/grey" if count == 0 else "frame_directory/orange"
                            os.makedirs(save_dir, exist_ok=True)
                            save_path = os.path.join(save_dir, f"cat_{current_time}.jpg")
                            cv2.imwrite(save_path, roi)
                            print(f"Saved cropped frame to: {save_path}")

                            # Draw the bounding box and label on the frame
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


                # Display the annotated frame
                cv2.imshow("Object Detection", frame)

                # Exit on pressing 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()

        count += 1

    cv2.destroyAllWindows()