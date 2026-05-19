import os
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras import layers, models

# --- Force CPU ---
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
mixed_precision.set_global_policy("float32")


from gated_cnn_model import *

# Paths
checkpoint_dir = 'models/'
output_dir = 'tflites/'

os.makedirs(output_dir, exist_ok=True)

fold = "2"

# Build and load weights
model = build_dual_stream_model(input_shape=(128, 4))
model.load_weights(checkpoint_dir + f'gated_cnn_smm_fold_{fold}.keras')

# Converter setup
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# DO NOT USE DEFAULT optimization (may upgrade op versions)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Force float input/output to avoid quantization issues
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

# Use only built-in ops
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

# Lower TensorList ops to keep TFLite-compatible
converter._experimental_lower_tensor_list_ops = True

# Convert and save
tflite_model = converter.convert()

output_path = output_dir + f'gated_cnn_smm_fold_{fold}.tflite'
with open(output_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved to: {output_path}")