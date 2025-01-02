import numpy as np
import cv2
import os

# Path to the folder containing the saved sequences

frame_width = 405
frame_height = 720
center_x = frame_width // 2
center_y = frame_height // 2


def draw_keypoints(frame, keypoints, color):
    """
    Draws keypoints on the given frame with the specified color.

    Parameters:
        frame (np.ndarray): The image frame to draw on.
        keypoints (np.ndarray): The keypoints to draw.
        color (tuple): The color to use for drawing the keypoints.
    """
    for point in keypoints:
        x,y = int(point[0]), int(point[1])
        print(x,y)
        cv2.circle(frame, ( x,y ), 5, color, -1)


def center_hip_to_frame(denormalized_frames_keypoints, frame):
    centered_frames_keypoints = []
    for frame_keypoints in denormalized_frames_keypoints:
        # Assuming the hip joint is the second keypoint
        hip_x, hip_y, = frame_keypoints[1]

        # Calculate offsets to center the hip joint at (center_x, center_y)
        offset_x = center_x - hip_x
        offset_y = center_y - hip_y

        # Apply offset to all keypoints
        centered_frame_keypoints = []
        for keypoint in frame_keypoints:
            centered_x = int(keypoint[0] * center_x/2) + offset_x
            centered_y = int(keypoint[1] * center_y/2) + offset_y
            centered_frame_keypoints.append([centered_x, centered_y])

        centered_frames_keypoints.append(centered_frame_keypoints)

    return centered_frames_keypoints


def visualize_sequences(sequence_file, video_path):
    # Load the sequences
    sequence = np.load(sequence_file)

    # Set up video writer to save the visualization
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30, (frame_width, frame_height))

    # Create a blank frame to draw on
    blank_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # Colors for visualization
    original_color = (0, 255, 0)  # Green for original sequence
    sequence_reshaped = sequence.reshape(sequence.shape[0], 6, 2)
    centered_keypoints = center_hip_to_frame(sequence_reshaped, blank_frame)
    print(sequence_reshaped.shape)
    # Visualize the original sequence
    for keypoints in centered_keypoints:
        frame = blank_frame.copy()
        draw_keypoints(frame, keypoints, original_color)  # Only using x and y for visualization
        out.write(frame)

    # Release the video writer
    out.release()
    print("Visualization video saved successfully.")


# Paths to the saved sequence files and the output video path
exercise = "squat"
video_file = "1.mp4"
epochs = 50
sequence_length = 60
sequence = 200
video_folder_path = f"Datasets/{exercise}/models/length_{sequence_length}/epochs_{epochs}/videos/{video_file.split('.')[0]}/sequences"
sequence_file = os.path.join(video_folder_path, f'{sequence}.original_sequence.npy')
interpolate_sequence_file = os.path.join(video_folder_path, f'{sequence}.interpolated_sequence.npy')
output_original_sequence_path = os.path.join(video_folder_path, f'{sequence}.original_sequence_visualization.mp4')
output_interpolated_sequence_path = os.path.join(video_folder_path, f'{sequence}.interpolated_sequence_visualization.mp4')


# Run the visualization
visualize_sequences(sequence_file, output_original_sequence_path)
visualize_sequences(interpolate_sequence_file, output_interpolated_sequence_path)
