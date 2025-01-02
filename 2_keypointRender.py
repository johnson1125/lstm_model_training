import cv2
import numpy as np
from utils import *


def draw_keypoints(video_path, keypoint_path, video_folder_path, i):
    # Load video and keypoints
    keypoints = np.load(keypoint_path)
    cap = cv2.VideoCapture(video_path)

    # Define output video writer
    out = cv2.VideoWriter(f"{video_folder_path}/{i}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30,
                          (int(cap.get(3)), int(cap.get(4))))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        height, width = frame.shape[:2]
        # Overlay keypoints on the frame
        keypoints_pixel = []
        for (x, y, z) in keypoints[frame_idx]:
            x_pixel = int(x * width)
            y_pixel = int(y * height)
            keypoints_pixel.append((x_pixel, y_pixel))

        # Draw lines between keypoints
        for i in range(len(keypoints_pixel) - 1):
            start_point = keypoints_pixel[i]
            end_point = keypoints_pixel[keypoint_link[i]]
            cv2.line(frame, start_point, end_point, (255, 0, 0), 2)  # Blue line with thickness 2

        for (x_pixel, y_pixel) in keypoints_pixel:
            cv2.circle(frame, (x_pixel, y_pixel), 5, (0, 255, 0), -1)

        # Write the frame
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(video_path + ' completed')

def process_folder_video(folder_path, label, video_count):
    video_folder_path = f"{folder_path}/videos/{label}/keypoints_drawed"
    create_directory_if_not_exists(video_folder_path)
    for i in range(1, video_count + 1):
        video_path = folder_path + f"/videos/{label}/{i}.mp4"
        keypoints_path = f"{folder_path}/keypoints/{label}/extracted_keypoints/{i}.npy"
        draw_keypoints(video_path, keypoints_path, video_folder_path, i+1)

def process_exercise(exercise, labels):
    for label in labels:
        folder_path = f"Datasets/{exercise}"
        video_count = count_videos_in_folder(folder_path, label)
        process_folder_video(folder_path, label, video_count)

# Input
exercise = "squat"
labels = ["proper_squat", "shallow_depth", "lifting_heel"]
keypoint_link= {0:1,1:2,2:3,3:4,4:5,5:3}

# exercise = "bicep_curl"
# labels = ["proper_bicep_curl", "elbow_drift", "incomplete_contraction","incomplete_extension","momentum_cheat"]
# keypoint_link= {0:1,1:0,2:0,3:2,4:3,5:3,6:3}

process_exercise(exercise, labels)