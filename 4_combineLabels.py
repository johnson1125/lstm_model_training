import os
import numpy as np
from utils import *



def count_npy_in_folder(folder_path, label,rotation):
    npy_folder_path = f'{folder_path}/keypoints/{label}/original/augmented/{rotation}'

    npy_count = 0

    # Iterate over all files in the directory
    for file_name in os.listdir(npy_folder_path):
        # Check if the file has one of the video extensions
        if os.path.splitext(file_name)[1].lower() == '.npy':
            npy_count += 1

    return npy_count

def process_data(folder_path, label, video_count):
    for i in range(1, video_count + 1):
        keypoint_path = f"{folder_path}/keypoints/{label}/extracted_keypoints/{i}.npy"
        label_path = f"{folder_path}/labeled_data/{label}/{i}.npy"
        x.append(np.load(keypoint_path))
        labels = np.load(label_path)
        y.append(labels)

def process_exercise(exercise, labels):
    for label in labels:
        folder_path = f"Datasets/{exercise}"
        video_count = count_videos_in_folder(folder_path, label)
        process_data(folder_path, label, video_count)



# Change Exercise
# Change Label
# exercise = "squat"
# labels = ["proper_squat", "shallow_depth", "lifting_heel"]
exercise = "bicep_curl"
labels = ["proper_bicep_curl", "elbow_drift", "incomplete_contraction", "momentum_cheat"]

x = []
y = []
combined_labeled_data_folder_path = f"Datasets/{exercise}/combined_labeled_data/"
create_directory_if_not_exists(combined_labeled_data_folder_path)

process_exercise(exercise, labels)

x_keypoints = np.concatenate(x, axis=0)
y_labels = np.concatenate(y, axis=0)


print(x_keypoints.shape)
print(y_labels.shape)


# Save the labeled data
np.save(combined_labeled_data_folder_path + "/X_data.npy", x_keypoints)
np.save(combined_labeled_data_folder_path + "/y_label.npy", y_labels)
print(f"{combined_labeled_data_folder_path}/y_label saved")
