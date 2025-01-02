import os
import tensorflow as tf


# exercise = "squat"
# labels = ["proper_squat", "shallow_depth", "lifting_heel"]
exercise = "bicep_curl"
labels = ["proper_bicep_curl", "elbow_drift", "incomplete_contraction", "momentum_cheat"]
normalization = "combined"
sequence_length = 60
model_selection = 6
epochs = 50
model_name = exercise + "_posture_model"
# model_name = "best_model"
output_dir =  f"Datasets/{exercise}/models/length_{sequence_length}/epochs_{epochs}"
file_path  = output_dir +"/"+ model_name + ".h5"

model = tf.keras.models.load_model(output_dir +"/"+ model_name + ".h5")

# Create a TFLiteConverter object
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable Select TF Ops
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,  # Built-in TFLite ops
                                       tf.lite.OpsSet.SELECT_TF_OPS]    # TensorFlow ops fallback

# Disable the experimental lowering of TensorList ops
converter._experimental_lower_tensor_list_ops = False
# Apply optimizations (optional but recommended)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_model = converter.convert()

# Save the converted model
with open(output_dir +"/"+ model_name + ".tflite", "wb") as f:
    f.write(tflite_model)
