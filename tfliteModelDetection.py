import os
import queue
import time
import threading
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from utils import *

def count_videos_in_folder(folder_path, video_extensions=['.mp4', '.avi', '.mkv', '.mov']):
    video_count = 0
    for file_name in os.listdir(folder_path):
        if os.path.splitext(file_name)[1].lower() in video_extensions:
            video_count += 1
    return video_count

# Function to count videos in the folder
# Load the trained model
exercise = "squat"
labels = ["proper_squat", "shallow_depth", "lifting_heel"]
# exercise = "bicep_curl"
# labels = ["proper_bicep_curl", "elbow_drift", "incomplete_contraction", "momentum_cheat"]

data = "keypoint_only"
sequence_length = 60
epochs = 50
model_name ="squat_posture_model"
method = "sliding_window"
mode = "recorded_video"  # real_time or recorded_video
camera_option=4
video_name = "1.mp4"
FIXED_WIDTH = 405
FIXED_HEIGHT = 720

video_folder_path = f"Datasets/{exercise}/models/length_{sequence_length}/epochs_{epochs}/videos/"

create_directory_if_not_exists(video_folder_path)

video_path = f"Datasets/{exercise}/test_videos/{video_name}"

new_video_name = str(count_videos_in_folder(video_folder_path) + 1)

tflite_model_path = f"Datasets/{exercise}/models/length_{sequence_length}/epochs_{epochs}/{model_name}.tflite"

interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose( min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# Initialize a list to store the sequence of keypoints
sequence = []
sequence_length = 60

def rel_position_normalize_keypoints(frame_keypoints):
    """
    Normalizes keypoints by translating the hip to the origin in 2D space.

    Args:
        frame_keypoints (np.ndarray): Array of keypoints with shape (6, 2).

    Returns:
        np.ndarray: Flattened array of normalized XY coordinates.
    """
    hip_x, hip_y = frame_keypoints[1][0], frame_keypoints[1][1]
    normalized_frame_keypoints = []
    for keypoint in frame_keypoints:
        normalized_x = keypoint[0] - hip_x
        normalized_y = keypoint[1] - hip_y
        normalized_frame_keypoints.append([normalized_x, normalized_y])
    return np.array(normalized_frame_keypoints).flatten()


def scale_and_rel_position_normalize_keypoints(frame_keypoints):
    """
    Normalizes and scales keypoints by scaling based on shoulder-hip distance
    and then translating the hip to the origin.

    Args:
        frame_keypoints (np.ndarray): Array of keypoints with shape (6, 2).

    Returns:
        np.ndarray: Flattened array of scaled and normalized XY coordinates.
    """
    hip_x, hip_y = frame_keypoints[1]
    shoulder_x, shoulder_y = frame_keypoints[0]

    # Step 1: Calculate the scaling factor
    hip = np.array([hip_x, hip_y])
    shoulder = np.array([shoulder_x, shoulder_y])
    scale = np.linalg.norm(hip - shoulder)
    scale_factor = 1 / scale if scale > 0 else 1.0  # Avoid division by zero

    # Step 2: Scale the keypoints
    scaled_frame_keypoints = []
    for keypoint in frame_keypoints:
        scaled_x = keypoint[0] * scale_factor
        scaled_y = keypoint[1] * scale_factor
        scaled_frame_keypoints.append([scaled_x, scaled_y])

    # Step 3: Translate the scaled hip to the origin
    hip_scaled_x, hip_scaled_y = scaled_frame_keypoints[1]
    normalized_frame_keypoints = []
    for keypoint in scaled_frame_keypoints:
        normalized_x = keypoint[0] - hip_scaled_x
        normalized_y = keypoint[1] - hip_scaled_y
        normalized_frame_keypoints.append([normalized_x, normalized_y])

    return np.array(normalized_frame_keypoints).flatten()
def extract_keypoints(results):
    """
    Extracts only the XY coordinates of selected keypoints from MediaPipe Pose results.

    Args:
        results: MediaPipe Pose results containing pose landmarks.

    Returns:
        np.ndarray: Flattened array of XY coordinates for selected keypoints.
    """
    landmarks = results.pose_landmarks.landmark
    if exercise == 'squat':
        keypoints = np.array([
            [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y],
            [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y],
            [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y],
            [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y],
            [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].x, landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].y],
            [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y]
        ])

    if exercise == 'bicep_curl':
        keypoints = np.array([
            [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y],
            [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y],
            [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y],
            [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y],
            [landmarks[mp_pose.PoseLandmark.RIGHT_PINKY].x, landmarks[mp_pose.PoseLandmark.RIGHT_PINKY].y],
            [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].x, landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].y],
            [landmarks[mp_pose.PoseLandmark.RIGHT_THUMB].x, landmarks[mp_pose.PoseLandmark.RIGHT_THUMB].y]
        ])




    return keypoints

def calculate_sequence_needed(current_fps, target_fps, sequence_length):
    # Duration of the sequence in seconds
    sequence_duration = sequence_length / target_fps

    # Number of frames needed from the real-time video
    sequence_needed = int(sequence_duration * current_fps + 0.5)  # +0.5 for rounding to nearest integer

    return sequence_needed

def interpolate_keypoints(frame1, frame2, num_interpolated_points):
    interpolated_frames = []

    # Convert frames to np.array for operations
    frame1 = np.array(frame1)
    frame2 = np.array(frame2)

    for i in range(num_interpolated_points):
        ratio = (i + 1) / (num_interpolated_points + 1)
        interpolated_frame = frame1 * (1 - ratio) + frame2 * ratio
        interpolated_frames.append(interpolated_frame)

    return interpolated_frames

def process_keypoints_sequence(keypoints_sequence, current_fps, target_fps, target_frame_count, frameNo):
    """
    Adjusts a sequence of keypoints to match the required frame count by interpolating frames.

    keypoints_sequence: List of keypoints arrays from the real-time video.
    current_fps: Frame rate at which keypoints are captured.
    target_fps: Desired frame rate for the LSTM model.
    target_frame_count: Number of frames needed for LSTM input.

    Returns: A list of keypoints arrays interpolated to match the target_frame_count,
    with the start and end frames being the same as the original sequence.
    """

    # Calculate the interpolation factor
    interpolation_factor = target_fps / current_fps

    # Create a list to store the interpolated keypoints
    interpolated_sequence = []

    # Track the indices of original frames
    original_frame_indices = []

    # Determine how many frames to include for each interval
    num_frames_between = int(interpolation_factor)

    # Interpolate between frames
    for i in reversed(range(1,len(keypoints_sequence))):
        # Append the original frame and store its index
        interpolated_sequence.insert(0,keypoints_sequence[i])
        original_frame_indices.append(len(interpolated_sequence) - 1)

        if i > 0 :
            num_interpolated_points = max(1,num_frames_between - 1)
            interpolated_frames = interpolate_keypoints(
                keypoints_sequence[i - 1], keypoints_sequence[i], num_interpolated_points)
            interpolated_sequence[:0] = interpolated_frames

    # Append the last frame and store its index
    interpolated_sequence.insert(0,keypoints_sequence[0])
    original_frame_indices.append(len(interpolated_sequence) - 1)

    # If the sequence is too long, remove non-original frames with gaps
    if len(interpolated_sequence) > target_frame_count:
        excess_frames = len(interpolated_sequence) - target_frame_count
        interpolated_sequence.reverse()
        # Create a list of non-original frame indices
        non_original_indices = [i for i in range(len(interpolated_sequence)) if i not in original_frame_indices]

        # # Remove non-original frames gradually with gaps
        # step = max(2, len(non_original_indices) // excess_frames)
        # frames_to_remove = sorted(non_original_indices[::step], reverse=True)[:excess_frames]
        frames_to_remove=[]
        # Repeat until the required number of frames to remove is reached
        while len(frames_to_remove) != excess_frames:
            # Select frames to remove using the current step
            step = max(2, len(non_original_indices) // (excess_frames - len(frames_to_remove)))
            additional_frames = sorted(non_original_indices[::step], reverse=True)[
                                :(excess_frames - len(frames_to_remove))]

            # Add selected frames to the list of frames to remove
            frames_to_remove.extend(additional_frames)

            # Remove selected frames from the list of non-original indices
            non_original_indices = [i for i in non_original_indices if i not in frames_to_remove]

        frames_to_remove = sorted(frames_to_remove,reverse=True)
        for i in frames_to_remove:
            del interpolated_sequence[i]

        interpolated_sequence.reverse()

    # If the sequence is too short, pad with interpolated frames
    elif len(interpolated_sequence) < target_frame_count:
        additional_frames_needed = target_frame_count - len(interpolated_sequence)

        step = max(2, len(interpolated_sequence) // additional_frames_needed)

        # Start from the last frame and move backward
        i = 0  # Start from the last frame
        while additional_frames_needed > 0:
            # Interpolate between the current frame and the previous frame
            if i < len(interpolated_sequence)-1 :  # Ensure there's a previous frame
                interpolated_frames = interpolate_keypoints(
                    interpolated_sequence[i], interpolated_sequence[i+1], 1)

                # Insert the interpolated frame before the current frame
                interpolated_sequence.insert(i, interpolated_frames[0])
                additional_frames_needed -= 1

            # Move the index backward by the step size
            i += step

            # Wrap around if we run out of indices to insert frames
            if i>=len(interpolated_sequence):
                i=0
    # Save the sequence (if needed)
    save_sequences(keypoints_sequence, interpolated_sequence, video_folder_path, frameNo)

    return interpolated_sequence

def save_sequences(sequence, interpolate_sequence, video_folder_path,frameNo):
    # Create directory for sequences if it does not exist
    sequences_dir = os.path.join(video_folder_path,new_video_name, 'sequences')
    create_directory_if_not_exists(sequences_dir)

    # Save sequences
    np.save(os.path.join(sequences_dir, str(frameNo) + '.original_sequence.npy'), np.array(sequence))
    np.save(os.path.join(sequences_dir, str(frameNo) + '.interpolated_sequence.npy'), np.array(interpolate_sequence))
    print("Sequences saved successfully.")

def predict_posture(sequence, interpreter):
    if len(sequence) == sequence_length:
        sequence_array = np.array(sequence).reshape(1, sequence_length, -1).astype(np.float32)

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], sequence_array)

        # Run inference
        interpreter.invoke()

        # Get the prediction
        output_data = interpreter.get_tensor(output_details[0]['index'])
        posture_class = np.argmax(output_data)
        confidence = np.max(output_data)
        return posture_class, confidence
    return None, None

def video_writer_thread(video_folder_path, new_video_name, frame_width, frame_height, fps, stop_event):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_folder_path + '/' + new_video_name + '.mp4', fourcc, fps, (frame_width, frame_height))

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1)
            out.write(frame)
        except queue.Empty:
            continue

    out.release()


rotate = False
if(mode == "real_time"):
    cap = cv2.VideoCapture(camera_option, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    rotate = True
else:
    cap = cv2.VideoCapture(video_path)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect_ratio = video_width / video_height
    print(video_width)
    print(video_height)

frame_width = FIXED_WIDTH
frame_height = FIXED_HEIGHT

# Queue for video writing
frame_queue = queue.Queue(maxsize=10)

# Event to signal thread stop
stop_event = threading.Event()

# Start video writer thread
writer_thread = threading.Thread(target=video_writer_thread,
                                 args=(video_folder_path, new_video_name, frame_width, frame_height, 30, stop_event))
writer_thread.start()

frame_count = 0
frame_skip = 2
start_time = time.time()
current_fps = 0
target_fps = 30
# Initialize a list to store the sequence of keypoints
sequence = []
interpolate_sequence = []
frame_no = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_no += 1

    if(rotate == True):
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    # Resize frame to the desired size
    frame = cv2.resize(frame, (FIXED_WIDTH, FIXED_HEIGHT))
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Keypoint extraction
    results = pose.process(image)
    if results.pose_landmarks:
        keypoints = extract_keypoints(results)
        keypoints_normalized = scale_and_rel_position_normalize_keypoints(keypoints)
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        sequence.append(keypoints_normalized)

    # Sequence interpolation
    sequence_needed = calculate_sequence_needed(current_fps, target_fps, sequence_length)
    if len(sequence) >= sequence_needed and current_fps != 0:
        interpolate_sequence = process_keypoints_sequence(sequence[-sequence_needed:], current_fps, target_fps,
                                                          sequence_length, frame_no)


    # Predict posture
    if len(sequence) > sequence_length:
        sequence.pop(0)
    posture_class, confidence = predict_posture(interpolate_sequence, interpreter)
    if posture_class is not None:
        cv2.putText(frame, f'Posture: {posture_class} (Confidence: {confidence:.2f})', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        current_fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()

    # Calculate the time taken for processing one frame in milliseconds
    time_per_frame_ms = (1 / current_fps) * 1000 if current_fps > 0 else 0

    # Write FPS and time per frame (in ms) on the frame
    cv2.putText(frame, f'Frame no: {frame_no}', (10, frame_height - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Write FPS and time per frame (in ms) on the frame
    cv2.putText(frame, f'FPS: {current_fps:.2f}', (10, frame_height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.putText(frame, f'Time per frame: {time_per_frame_ms:.2f} ms', (10, frame_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    frame_queue.put(frame)

    # Display the resulting frame
    cv2.imshow('Posture Monitoring', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

stop_event.set()
writer_thread.join()

cap.release()
cv2.destroyAllWindows()
