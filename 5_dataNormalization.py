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
def scale_and_rel_position_normalize_keypoints(frames_keypoints):
    counter1 = 0
    counter2 = 0
    normalized_frames_keypoints = []
    for frame_keypoints in frames_keypoints:
        # Step 1: Scaling Normalization
        # Calculate the scale using the distance between the hip and shoulder
        hip_x, hip_y, hip_z = frame_keypoints[1][0], frame_keypoints[1][1], frame_keypoints[1][2]
        hip = np.array([hip_x, hip_y])
        shoulder = np.array([frame_keypoints[0][0], frame_keypoints[0][1]])
        scale = np.linalg.norm(hip - shoulder)
        scale_factor = 1 / scale if scale > 0 else 1.0  # Avoid division by zero

        # Scale each keypoint
        scaled_frame_keypoints = []
        for keypoint in frame_keypoints:
            scaled_x = keypoint[0] * scale_factor
            scaled_y = keypoint[1] * scale_factor
            scaled_z = keypoint[2] * scale_factor
            scaled_frame_keypoints.append([scaled_x, scaled_y, scaled_z])

        # Step 2: Relative Position Normalization
        # Use the hip (index 1) as the reference point after scaling
        hip_x_scaled, hip_y_scaled, hip_z_scaled = scaled_frame_keypoints[1][0], scaled_frame_keypoints[1][1], \
        scaled_frame_keypoints[1][2]
        rel_position_normalized_frame = []
        for keypoint in scaled_frame_keypoints:
            normalized_x = keypoint[0] - hip_x_scaled
            normalized_y = keypoint[1] - hip_y_scaled
            normalized_z = keypoint[2] - hip_z_scaled
            rel_position_normalized_frame.append([normalized_x, normalized_y, normalized_z])

        # Append the normalized keypoints
        normalized_frames_keypoints.append(np.array(rel_position_normalized_frame).flatten())
        if scale > 0:
            counter1 += 1
        else:
            counter2 += 1

    print("Number of valid frames normalized with scaling:", counter1)
    print("Number of frames skipped due to zero scale:", counter2)
    return np.array(normalized_frames_keypoints)
def combine_normalize_keypoints(folder_path, label, video_count):

    for i in range(1, video_count + 1):
        keypoint_path = f"{folder_path}/keypoints/{label}/extracted_keypoints/{i}.npy"
        keypoints = np.load(keypoint_path)

        # Apply combined scale and relative position normalization
        normalized_x_combined = scale_and_rel_position_normalize_keypoints(keypoints)
        x_combined.append(normalized_x_combined)

def process_exercise(exercise, labels):
    for label in labels:
        folder_path = f"Datasets/{exercise}"
        video_count = count_videos_in_folder(folder_path, label)
        combine_normalize_keypoints(folder_path, label, video_count)


# Change Exercise
# Change Label
# exercise = "squat"
# labels = ["proper_squat", "shallow_depth", "lifting_heel"]
exercise = "bicep_curl"
labels = ["proper_bicep_curl", "elbow_drift", "incomplete_contraction", "momentum_cheat"]

x_combined = []
process_exercise(exercise, labels)

x_combined_normalized = np.concatenate(x_combined, axis=0)

print(x_combined_normalized.shape)

normalized_Data_folder_path = f"Datasets/{exercise}/normalized_Data/"

create_directory_if_not_exists(normalized_Data_folder_path)

# Save normalized data
np.save(normalized_Data_folder_path + "/X_keypoints_normalized.npy", x_combined_normalized)

print(normalized_Data_folder_path  + "X_keypoints_normalized.npy saved")
