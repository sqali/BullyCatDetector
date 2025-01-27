from augmentation import augment_and_save_images
from frame_cluster_analysis import frame_pipeline, store_pixel_info_with_labels, create_pca, visualize_pca, logistic_regression_training


# Create augmented images of my cats and the bully cat
augment_and_save_images()

# Cluster Frame Analysis and model creation
frame_pipeline()
store_pixel_info_with_labels()
create_pca()
visualize_pca()
logistic_regression_training()