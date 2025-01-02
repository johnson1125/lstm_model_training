import io
import numpy as np
import pandas as pd
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential,load_model
from sklearn.model_selection import train_test_split
import seaborn as sns
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization
from keras.callbacks import EarlyStopping
from utils import *

def save_model_summary(model, file_path):
    # Capture the model summary as a string
    with io.StringIO() as stream:
        model.summary(print_fn=lambda x: stream.write(x + '\n'))  # Redirect summary to StringIO
        summary_str = stream.getvalue()  # Get the content of the StringIO object

    # Save the summary string to a file
    with open(file_path, 'w') as f:
        f.write(summary_str)


def plot_classification_report(y_test, y_pred_classes, model_name, output_dir):
    report = classification_report(y_test, y_pred_classes, output_dict=True)
    df_report = pd.DataFrame(report).transpose()

    plt.figure(figsize=(10, 6))
    sns.heatmap(df_report.iloc[:-1, :].T, annot=True, cmap="Blues", cbar=False, fmt=".2f")
    plt.title("Classification Report Heatmap")
    plt.savefig(f'{output_dir}/{model_name}_classificationReport.png')
    plt.show()


def plot_confusion_matrix(y_test, y_pred_classes, labels, model_name, output_dir):
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'{output_dir}/{model_name}_confusionMatrix.png')
    plt.show()


def plot_loss(history, model_name, output_dir):
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.savefig(f'{output_dir}/{model_name}_loss_vs_epoch.png')
    plt.show()

def load_data(X_path, y_path):
    """
    Load the dataset and extract only XY coordinates from XYZ data.

    Parameters:
    - X_path: str, path to the numpy file containing keypoint sequences.
    - y_path: str, path to the numpy file containing labels.

    Returns:
    - X: np.ndarray, shape (samples, sequence_length, keypoints * 2)
    - y: np.ndarray, shape (samples,)
    """
    X = np.load(X_path)  # Original shape: (samples, sequence_length, keypoints * 3)
    y = np.load(y_path)

    # Determine the number of keypoints
    num_features = X.shape[2]
    if num_features % 3 != 0:
        raise ValueError("The number of features in X is not divisible by 3. Check the data format.")

    num_keypoints = num_features // 3

    # Reshape to (samples, sequence_length, keypoints, 3)
    X = X.reshape(-1, X.shape[1], num_keypoints, 3)

    # Extract only XY coordinates
    X = X[:, :, :, :2]  # Now shape: (samples, sequence_length, keypoints, 2)

    # Reshape back to (samples, sequence_length, keypoints * 2)
    X = X.reshape(-1, X.shape[1], num_keypoints * 2)

    return X, y

def split_data(X, y, test_size=0.2):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def lstm_model(X):
    model = Sequential()
    model.add(
        LSTM(64, return_sequences=True, activation='tanh', input_shape=(X.shape[1], X.shape[2]), recurrent_dropout=0.2))
    model.add(LayerNormalization())
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=True, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=False, activation='tanh'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001, clipvalue=1.0), loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

# Main code to run
# exercise = "squat"
# labels = ["proper_squat", "shallow_depth", "lifting_heel"]
exercise = "bicep_curl"
labels = ["proper_bicep_curl", "elbow_drift", "incomplete_contraction", "momentum_cheat"]
sequence_length = 60
epochs = 50
model_name = f"{exercise}_posture_model"
output_dir = f"Datasets/{exercise}/models/length_{sequence_length}/epochs_{epochs}"
create_directory_if_not_exists(output_dir)

X, y = load_data(f"Datasets/{exercise}/sequence_data/{sequence_length}/X_sequences.npy",
                 f"Datasets/{exercise}/sequence_data/{sequence_length}/y_sequence_labels.npy")

X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

model = lstm_model(X)

summary_file_path = f"{output_dir}/{model_name}_summary.txt"
save_model_summary(model, summary_file_path)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), batch_size=32, verbose=1, callbacks=[])

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Evaluate the model and make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Plot and save reports
plot_classification_report(y_test, y_pred_classes, model_name, output_dir)
plot_confusion_matrix(y_test, y_pred_classes, labels, model_name, output_dir)
plot_loss(history, model_name, output_dir)

# Save the trained model
model.save(f"{output_dir}/{model_name}.h5")
