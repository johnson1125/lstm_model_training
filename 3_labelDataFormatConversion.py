import os
import xml.etree.ElementTree as ET
import numpy as np

def convert_xml_to_npy(xml_file_path, npy_file_path):
    # Load and parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Determine the total number of frames (from metadata)
    start_frame = int(root.find('meta/job/start_frame').text)
    stop_frame = int(root.find('meta/job/stop_frame').text)
    total_frames = stop_frame - start_frame + 1

    # Initialize the NumPy array with zeros for each frame
    annotations_np = np.zeros((total_frames, 1), dtype=int)



    # Loop through each track to populate the array
    for track in root.findall('track'):
        label = track.get('label')
        label_index = label_indices.get(label)

        if label_index is not None:
            for box in track.findall('box'):
                frame = int(box.get('frame'))
                annotations_np[frame] = label_index

    # Save the annotation data to an .npy file
    np.save(npy_file_path, annotations_np)
    print(f"{xml_file_path} completed")
def process_folder_xml(folder_path, label):
    # Path to the XML label files for the specified label
    label_folder_path = os.path.join(folder_path, "labeled_data", label)


    try:
        # Iterate over all XML files in the directory
        for xml_file_name in os.listdir(label_folder_path):
            if xml_file_name.endswith('.xml'):
                xml_file_path = os.path.join(label_folder_path, xml_file_name)
                npy_file_name = os.path.splitext(xml_file_name)[0] + '.npy'
                npy_file_path = os.path.join(label_folder_path, npy_file_name)
                # Convert XML to NumPy array and save
                convert_xml_to_npy(xml_file_path, npy_file_path)
    except FileNotFoundError:
            print(f"{label_folder_path} not found")
            return


def process_exercise(exercise, labels):
    # Process each label within the exercise
    for label in labels:
        folder_path = f"Datasets/{exercise}"
        process_folder_xml(folder_path, label)


# Change the exercise and labels accordingly
# exercise = "squat"
# labels = ["proper_squat", "shallow_depth", "lifting_heel"]
# label_indices = {'proper_squat': 0, 'lifting_heels': 1, 'shallow_depth': 2}

exercise = "bicep_curl"
labels = ["proper_bicep_curl", "elbow_drift", "incomplete_contraction","momentum_cheat"]
label_indices = {'proper_bicep_curl': 0, 'elbow_drift': 1, 'incomplete_contraction': 2,'momentum_cheat': 3}


# Start processing
process_exercise(exercise, labels)
