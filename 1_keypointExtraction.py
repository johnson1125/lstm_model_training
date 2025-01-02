import cv2
import mediapipe as mp
import numpy as np
from utils import *

# Initialize MediaPipe pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose( model_complexity=1)
mp_drawing = mp.solutions.drawing_utils
FIXED_WIDTH = 405
FIXED_HEIGHT = 720

# Function to extract key points from a single frame and normalize them
def extract_keypoints(results,exercise):
    # Extract right-side landmarks

    if exercise=="squat":
        # squat keypoints
        visible_landmarks = [
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER],
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP],
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE],
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE],
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL],
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
        ]

    if exercise == "bicep_curl":
    # bicep_curl keypoints
        visible_landmarks = [
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER],
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP],
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW],
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST],
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY],
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX],
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB],
        ]

    # Extract keypoints
    keypoints = [[landmark.x, landmark.y, landmark.z] for landmark in visible_landmarks]
    keypoints_3d = np.array(keypoints)

    result_keypoints  = [[landmark.x, landmark.y, landmark.z] for landmark in results.pose_landmarks.landmark]
    result_keypoints_3d = np.array(result_keypoints)

    return keypoints_3d, result_keypoints_3d


# Function to process video and extract key points for all frames
def process_video(video_path,exercise):
    cap = cv2.VideoCapture(video_path)
    extracted_keypoints_list = []
    results_keypoints_list = []
    frame_id = 0
    pose_detect = 0
    pose_not_detect = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (FIXED_WIDTH, FIXED_HEIGHT))
        # Convert the frame to RGB and resize
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the frame with MediaPipe Pose
        results = pose.process(image)

        if results.pose_landmarks:
            # Extract and normalize keypoints
            keypoints_3d, result_keypoints_3d= extract_keypoints(results,exercise)
            extracted_keypoints_list.append(keypoints_3d)
            results_keypoints_list.append(result_keypoints_3d)
            pose_detect += 1
        else:
            pose_not_detect += 1

        frame_id += 1

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Pose', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(video_path + " completed")
    return  np.array(extracted_keypoints_list), np.array(results_keypoints_list)

# Function to process all videos in a folder and save keypoints
def process_folder_video(folder_path, label, video_count,exercise):
    original_Keypoint_folder_dir = f"{folder_path}/keypoints/{label}/extracted_keypoints"
    results_Keypoint_folder_dir = f"{folder_path}/keypoints/{label}/results"
    create_directory_if_not_exists(original_Keypoint_folder_dir)
    create_directory_if_not_exists(results_Keypoint_folder_dir)

    for i in range(1, video_count + 1):
        video_path = f"{folder_path}/videos/{label}/{i}.mp4"
        keypoints_list,results_keypoints_list = process_video(video_path,exercise)
        np.save(f"{original_Keypoint_folder_dir}/{i}.npy", keypoints_list)
        np.save(f"{results_Keypoint_folder_dir}/{i}.npy", results_keypoints_list)

# Process a specific exercise by labels
def process_exercise(exercise, labels):
    for label in labels:
        folder_path = f"Datasets/{exercise}"
        video_count = count_videos_in_folder(folder_path, label)
        process_folder_video(folder_path, label, video_count,exercise)


# Change the exercise and labels accordingly
# exercise = "squat"
# labels = ["proper_squat", "shallow_depth", "lifting_heel"]
exercise = "bicep_curl"
labels = ["proper_bicep_curl", "elbow_drift", "incomplete_contraction","momentum_cheat"]

# Start processing the exercise videos
process_exercise(exercise, labels)
