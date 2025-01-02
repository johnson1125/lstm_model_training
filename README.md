## Workout Posture Monitoring Model Training

The Workout Posture Monitoring Model Training Project is a key component of the real-time workout posture monitoring system, designed to classify workout postures and provide real-time feedback to users. The project involves the following stages:

1. **Dataset Preparation:**
   - Videos capturing exercises like squats and bicep curls are preprocessed to extract keypoints using MediaPipe, ensuring accurate representation of body movements.
   - Variations in postures are labeled to distinguish correct and incorrect techniques.

2. **Data Preprocessing:**
   - Keypoints are normalized using scaling and relative position normalization to standardize input data.
   - Sequences are generated using a sliding window approach to capture temporal dependencies.

3. **Model Training:**
   - A sequential LSTM model is trained to classify postures based on these sequences.
   - The model is designed with dropout layers to prevent overfitting and is optimized for accuracy using labeled data.

4. **Model Conversion:**
   - The trained model is converted into TensorFlow Lite format for efficient real-time deployment, ensuring smooth inference performance.

5. **Integration and Feedback:**
   - The model is integrated into the posture monitoring system to analyze live video feeds, predict postures, and provide actionable feedback in real-time.

## Prerequisites

Before setting up the environment, ensure the following tools are installed on your system:

1. Anaconda
2. Python 3.8 or above
3. GPU with CUDA support

## Setting Up the Environment

1. Create a new Conda environment using the provided `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   ```

2. Activate the environment:
   ```bash
   conda activate training_env
   ```

[Learn more about Anaconda](https://www.anaconda.com/)

## Functionalities of Each Script

1. **`1_keypointExtraction.py`:**
   - Extracts keypoints from video frames using MediaPipe Pose.
   - Outputs normalized keypoints for each exercise.

2. **`2_keypointRender.py`:**
   - Overlays extracted keypoints on the video for visualization.
   - Saves videos with keypoints for easier data verification and labeling.

3. **`3_labelDataFormatConversion.py`:**
   - Converts labeled XML annotations into `.npy` format for machine learning workflows.
   - Ensures compatibility with training pipelines.

4. **`4_combineLabels.py`:**
   - Combines labels from different datasets into a unified format.
   - Prepares concatenated datasets for training.

5. **`5_dataNormalization.py`:**
   - Normalizes keypoints using scaling and relative position techniques.
   - Saves normalized keypoints for downstream machine learning tasks.

6. **`6_sequenceGeneration.py`:**
   - Generates sequences from normalized keypoints.
   - Creates overlapping sequences to preserve temporal dynamics.

7. **`7_modelTraining.py`:**
   - Defines and trains the LSTM model for posture classification.
   - Saves trained models and generates evaluation reports.

8. **`8_modelConverter.py`:**
   - Converts the trained TensorFlow model to TensorFlow Lite format.
   - Optimizes the model for real-time inference.

9. **`tfliteModelDetection.py`:**
   - Implements real-time posture detection using the TFLite model.
   - Provides visual and textual feedback on detected postures.

10. **`sequenceVisualization.py`:**
    - Visualizes sequences of keypoints for debugging and verification.
    - Generates videos highlighting the sequences used in training.
