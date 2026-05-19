import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
from scipy.interpolate import CubicSpline
import re
import random
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from scipy.signal import savgol_filter
from filterpy.kalman import KalmanFilter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, Callback
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.layers import Add, Dense, LayerNormalization, GlobalAveragePooling1D, Conv1D, Dropout, MultiHeadAttention, Layer, Embedding, Concatenate
from tensorflow.keras.initializers import TruncatedNormal

from tensorflow.keras.layers import RNN, SimpleRNNCell

from sklearn.metrics import precision_recall_fscore_support, classification_report, roc_auc_score, roc_curve, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy import signal

import pywt
import cv2

import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MaxPooling1D, LSTM, add, BatchNormalization
from sklearn.utils.class_weight import compute_class_weight

import gc

from scipy.signal import welch, find_peaks
from scipy.stats import entropy
#--------------------------------------
#%% Model Gated CNN
# -----------------------------------------------
# 1. GatedCNN: Python 3.12-Compatible
# -----------------------------------------------
class GatedCNN(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super(GatedCNN, self).__init__(**kwargs)
        self.conv = layers.Conv1D(dim, kernel_size=1, activation='gelu')
        self.gate = layers.Dense(dim, activation='sigmoid')
        self.proj = layers.Dense(dim)

    def call(self, x):
        x_proj = self.conv(x)
        gate = self.gate(x_proj)
        x_gated = x_proj * gate
        return self.proj(x_gated)

    def get_config(self):
        config = super(GatedCNN, self).get_config()
        config.update({"dim": self.conv.filters})
        return config

# -----------------------------------------------
# 2. CNN Feature Extractor for Each Stream
# (Preserves sequence output)
# -----------------------------------------------
def cnn_feature_extractor_with_sequence_output(input_shape=(128, 3)):
    return models.Sequential([
        layers.Conv1D(32, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),  # -> (64, 32)
        layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        layers.MaxPooling1D(pool_size=2),  # -> (32, 64)
        layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'),
        # Final shape: (batch, 32, 64)
    ])

# -----------------------------------------------
# 3. Classifier Head
# -----------------------------------------------
def binary_classifier_head():
    return models.Sequential([
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.Dense(1, activation='sigmoid')
    ])

# -----------------------------------------------
# 4. Dual-Stream Model with gatedCNN
# -----------------------------------------------
def build_dual_stream_model(input_shape=(128, 3)):
    input1 = layers.Input(shape=input_shape, name='accelerometer_input')
    input2 = layers.Input(shape=input_shape, name='gyroscope_input')

    # CNN outputs with sequence dimension preserved
    cnn1 = cnn_feature_extractor_with_sequence_output(input_shape)(input1)
    cnn2 = cnn_feature_extractor_with_sequence_output(input_shape)(input2)

    # Gated temporal modeling
    gatedCNN_out1 = GatedCNN(dim=32)(cnn1)
    gatedCNN_out2 = GatedCNN(dim=32)(cnn2)

    # Global Average Pooling to get fixed-length vector
    pooled1 = layers.GlobalAveragePooling1D()(gatedCNN_out1)
    pooled2 = layers.GlobalAveragePooling1D()(gatedCNN_out2)

    # Concatenate and classify
    concatenated = layers.Concatenate()([pooled1, pooled2])  # Shape: (batch, 128)
    output = binary_classifier_head()(concatenated)

    return models.Model(inputs=[input1, input2], outputs=output)
#--------------------------------------
