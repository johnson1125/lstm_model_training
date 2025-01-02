import os

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory {directory} created.")
    else:
        print(f"Directory {directory} already exists.")

def count_videos_in_folder(folder_path, label, video_extensions=['.mp4', '.avi', '.mkv', '.mov']):
    """
    Count the number of video files in a folder.

    Args:
        folder_path (str): Path to the folder containing video files.
        video_extensions (list): List of video file extensions to look for.

    Returns:
        int: Number of video files in the folder.
    """
    video_folder_path = f"{folder_path}/videos/{label}"

    video_count = 0

    # Iterate over all files in the directory
    for file_name in os.listdir(video_folder_path):
        # Check if the file has one of the video extensions
        if os.path.splitext(file_name)[1].lower() in video_extensions:
            video_count += 1

    return video_count