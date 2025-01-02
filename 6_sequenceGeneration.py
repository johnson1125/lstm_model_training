import numpy as np
from utils import *

def create_sequences(data, labels, sequence_length):
    sequences = []
    sequence_labels = []

    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
        sequence_labels.append(labels[i + sequence_length - 1])

    return np.array(sequences), np.array(sequence_labels)

# Load the normalized data and labels
# exercise = "squat"
exercise = "bicep_curl"

sequence_length = 60
output_dir = f"Datasets/{exercise}/sequence_data/{sequence_length}"
create_directory_if_not_exists(output_dir)
X_normalized = np.load(f'Datasets/{exercise}/normalized_Data/X_keypoints_normalized.npy')
y = np.load(f'Datasets/{exercise}/combined_labeled_data/y_label.npy')

# Create sequences from the data
X_sequences, y_sequences = create_sequences(X_normalized, y, sequence_length)

print(X_normalized.shape)
print(y.shape)
print(X_sequences.shape)
print(y_sequences.shape)

# Save the sequences
np.save(f"Datasets/{exercise}/sequence_data/{sequence_length}/X_sequences.npy", X_sequences)
np.save(f"Datasets/{exercise}/sequence_data/{sequence_length}/y_sequence_labels.npy", y_sequences)

print(f"Datasets/{exercise}/sequence_data/{sequence_length}/y_sequence_labels.npy saved")