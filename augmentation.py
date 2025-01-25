# Standard library imports
import os
from datetime import datetime

# Third-party library imports
import cv2
import albumentations as A


def augment_and_save_images():

    """
    Function to augment the images in the input folder and save them in the output folder

    Args:
        None

    Returns:
        None

    Raises:
        None
    """

    image_folders = ["frame_directory/grey","frame_directory/orange"]

    for folder in image_folders:
        input_folder = folder
        output_folder = input_folder

        # Define a refined augmentation pipeline
        transform = A.Compose([
                        A.Rotate(limit=45, p=0.5),  # Limit rotation to a smaller range
                        A.HorizontalFlip(p=0.5),  # Horizontal flip
                        A.VerticalFlip(p=0.3),  # Vertical flip with lower probability
                        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),  # Small shift, scale, and rotation
                        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),  # Limited brightness and contrast changes
                        A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),  # Lower noise variance
                        A.Blur(blur_limit=1, p=0.4),  # Minimal blur
                        A.RandomSizedCrop(min_max_height=(120, 180), height=256, width=256, p=0.5),  # Controlled cropping
                        A.ElasticTransform(alpha=0.8, sigma=20, p=0.5),  # Controlled elastic transform
                        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),  # Controlled hue/saturation shifts
                        A.CoarseDropout(max_holes=4, max_height=8, max_width=8, p=0.3)  # Reduced dropout
                    ])

            # Example usage
            # transformed = transform(image=image)
            # augmented_image = transformed['image']


        
        for filename in os.listdir(input_folder):
            if filename.endswith("jpg") or filename.endswith("png"):
                img_path = os.path.join(input_folder, filename)
                img = cv2.imread(img_path)

                for i in range(10): # Loop to create 10 augmented images for each image
                    augmented = transform(image=img) # Outputs a dictionary with the augmented image and metadata
                    augmented_image = augmented["image"] # augmented["image"] contains the augmented image

                    if "grey" in folder:
                        output_image_path = os.path.join(output_folder, f"{filename[:-4]}_{i}_grey.jpg")
                    else:
                        output_image_path = os.path.join(output_folder, f"{filename[:-4]}_{i}_orange.jpg")

                    cv2.imwrite(output_image_path, augmented_image)

        print("Augmentation Completed Successfully for folder: ", folder)

augment_and_save_images()