#%% Load libraries
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

np.random.seed(42)

#%% Helper functions
from scipy.stats import entropy

def compute_orientation(data, eps=1e-8):
    """
    data: (N, 128, 3) → [ax, ay, az]
    return: (N, 128, 2) → [pitch, roll]
    """
    print("Computing orientation (pitch, roll)...")
    ax = data[..., 0]
    ay = data[..., 1]
    az = data[..., 2]
    # Prevent division by zero
    denom = np.sqrt(ay**2 + az**2) + eps
    pitch = np.arctan2(-ax, denom)        # (N,128)
    roll  = np.arctan2(ay, az + eps)      # (N,128)
    # Stack to shape (N,128,2)
    orientation = np.stack([pitch, roll], axis=-1)
    return orientation

def compute_entropy_feature(data, eps=1e-8):
    print("Computing entropy feature...")
    # Absolute values to avoid negative probabilities
    abs_data = np.abs(data) + eps  # (N, 128, 3)
    # Normalize across axis dimension → convert (x,y,z) into a prob distribution
    probs = abs_data / np.sum(abs_data, axis=2, keepdims=True)
    # Compute entropy per timestep
    ent = entropy(probs, axis=2)  # (N, 128)

    return ent[..., np.newaxis]   # (N, 128, 1)

def compute_magnitude(data):
    """
    data: (N, 128, 3)
    return: (N, 128, 1)
    """
    print("Computing magnitude feature...")
    mag = np.sqrt(np.sum(data**2, axis=2))
    return mag[..., np.newaxis]


def compute_feature_gradient(data):
    print("Computing gradient feature...")
    diff = np.diff(data, axis=1)
    grad = np.linalg.norm(diff, axis=2)
    grad = np.pad(grad, ((0,0),(1,0)), mode='constant')
    return grad[..., np.newaxis]

def compute_window_gradient(windows):
    print("Computing window gradient feature...")
    gradient_maps = []
    for window in windows:
        gradient_map = np.gradient(window, axis=0)  # → (128,3)
        gradient_maps.append(gradient_map)
    return np.array(gradient_maps)

def compute_tilt_change(data, eps=1e-8):
    """
    data: (N,128,3)
    return: (N,128,1)  change in gravity direction between timesteps
    """
    print("Computing tilt change feature...")
    # Normalize accel to gravity direction
    norm = np.sqrt(np.sum(data**2, axis=2, keepdims=True)) + eps
    g = data / norm   # (N,128,3)
    # Compute dot product between consecutive gravity vectors
    dot = np.sum(g[:,1:,:] * g[:,:-1,:], axis=2)  # (N,127)
    # Clamp for numerical safety
    dot = np.clip(dot, -1.0, 1.0)
    # Angle between vectors
    dtheta = np.arccos(dot)  # (N,127)
    # Pad first timestep
    dtheta = np.pad(dtheta, ((0,0),(1,0)), mode='constant')
    return dtheta[..., np.newaxis]  # (N,128,1)

def compute_gravity_direction_variance(data, eps=1e-8):
    print("Computing gravity direction variance feature...")
    norm = np.sqrt(np.sum(data**2, axis=2, keepdims=True)) + eps
    g = data / norm  # (N,128,3)

    var = np.var(g, axis=1)     # (N,3)
    total_var = np.sum(var, axis=1)  # (N,)
    return total_var[:, np.newaxis, np.newaxis]   # (N,1,1)

def compute_jerk(data):
    print("Computing jerk feature...")
    diff = np.diff(data, axis=1)   # (N,127,3)
    jerk = np.linalg.norm(diff, axis=2)  # (N,127)
    jerk = np.pad(jerk, ((0,0),(1,0)), mode="constant")
    return jerk[..., np.newaxis]   # (N,128,1)

def add_features(data):
    mag = compute_magnitude(data)
    # orient = compute_orientation(data) # (N,128,2)
    # tilt  = compute_tilt_change(data) # (N,128,1)
    # jerk  = compute_jerk(data)                       # (N,128,1)
    # gvar  = compute_gravity_direction_variance(data) # (N,1,1)
    # gvar  = np.repeat(gvar, 128, axis=1)             # → (N,128,1)
    
    # ent = compute_entropy_feature(data)
    # grad = compute_feature_gradient(data)
    return np.concatenate([data, mag], axis=-1) # grad, tilt, jerk, gvar

# Function to compute the gradient for each window
def compute_gradient(windows):
    gradient_maps = []
    for window in windows:
        # Compute the gradient (change between consecutive time steps) for each feature (x, y, z)
        gradient_map = np.gradient(window, axis=0)  # gradient along time axis
        gradient_maps.append(gradient_map)
    return np.array(gradient_maps)

# Function to sort the window scores and keep track of the window index
def sort_window_scores(window_scores):
    # Sort the scores in descending order and keep the indices
    sorted_indices = np.argsort(window_scores)[::-1]  # [::-1] to reverse for descending order
    sorted_scores = window_scores[sorted_indices]
    return sorted_scores, sorted_indices

# Function to score windows based on gradient
def score_windows_by_gradient(optical_flow_maps):
    scores = []
    for flow_map in optical_flow_maps:
        # Compute the sum of absolute gradients across all time steps and features (x, y, z)
        score = np.sum(np.abs(flow_map))
        scores.append(score)
    return np.array(scores)

#--------------------------------------
# Function to parse the filename
def parse_filename(filename):
    match = re.match(r'S(\d+)A(\d+)T(\d+)\.csv', filename)
    if match:
        subject = int(match.group(1))
        activity = int(match.group(2))
        round = int(match.group(3))
        return subject, activity, round
    return None, None, None

# Function to label based on activity number
def label_activity(activity):
    if 10 <= activity <= 14:
        return 1
    else:
        return 0
    
def handle_unexpected_columns(file_path):
    # Handle cases with unexpected number of columns by picking the last three columns
    df = pd.read_csv(file_path, sep=';', header=None)
    
    if df.shape[1] < 3:
        raise ValueError("CSV file does not contain enough columns.")
    # Pick the last three columns
    return df.iloc[:, -3:]

def fix_faulty_csv(file_path):
    try:
        df = pd.read_csv(file_path, header=None)
        
        # Attempt to fix the CSV by converting non-numeric values to NaN
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna(how='any', axis=0)
        
        # Handle cases based on the number of columns
        if len(df.columns) == 4:
            df = df.iloc[:, 1:4]  # Skip the first column with timestamps
        elif len(df.columns) == 3:
            df = df.iloc[:, 0:3]  # Take all columns if there are only 3
        else:
            raise ValueError("Unexpected number of columns")
        
        # Rename columns
        df.columns = ['x', 'y', 'z']
        
        # Data cleaning steps
        df = df.astype(float)
        mask = df.isnull().all(axis=1)
        df = df.drop(df[mask].index)
        df = df.ffill().bfill()
        
        print(f"Fixed file {file_path}")
        return df
    except Exception as e:
        print(f"Failed to fix file {file_path}: {e}")
        return None
    
# Function to process the CSV data and clean it (as before)
def process_csv(file_path, denoise=False):
    try:
        df = pd.read_csv(file_path, header=None)
        if len(df.columns) == 4:
            df = df.iloc[:, 1:4]  # Skip the first column with timestamps
        elif len(df.columns) == 3:
            df = df.iloc[:, 0:3]  # Take all columns if there are only 3
        else:
            # Call the function to handle unexpected number of columns
            df = handle_unexpected_columns(file_path)
        df.columns = ['x', 'y', 'z']
        
        # Data cleaning steps
        df = df.astype(float)
        mask = df.isnull().all(axis=1)
        df = df.drop(df[mask].index)
        df = df.ffill().bfill()
        
        return df
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return fix_faulty_csv(file_path)

# Function to handle sliding windows (as before)
def sliding_window(data, label, clearing_time_index, max_time, sub_window_size, stride_size):
    assert clearing_time_index >= sub_window_size - 1, "Clearing value needs to be greater or equal to (window size - 1)"
    start = clearing_time_index - sub_window_size + 1 

    if max_time >= data.shape[0] - sub_window_size:
        max_time = max_time - sub_window_size + 1

    sub_windows  = (
        start + 
        np.expand_dims(np.arange(sub_window_size), 0) + 
        np.expand_dims(np.arange(max_time, step=stride_size), 0).T
    )
    data_windows = data[sub_windows]
    label_windows = label[sub_windows]
    return data_windows, label_windows

#--------------------------------------
def parse_csv_filenames(base_path):
    files = [f for f in os.listdir(base_path) if f.endswith('.csv')]

    # Regex pattern to extract subject, activity, and trial
    pattern = re.compile(r"S(\d+)A(\d+)T(\d+)")

    # Parse filenames
    records = []
    for file in files:
        match = pattern.search(file)
        if match:
            records.append((file, int(match.group(1)), int(match.group(2)), int(match.group(3))))

    df = pd.DataFrame(records, columns=['Filename', 'Subject', 'Activity', 'Trial'])
    return df

def split_subjects(base_df, val_size, test_size):
    subjects = sorted(base_df['Subject'].unique())
    np.random.shuffle(subjects)
    print(subjects)
    
    t_rm = [28, 29, 30, 31, 33, 35, 38, 39, 32, 36, 37, 43, 44, 45, 46, 49, 51, 56, 57, 58, 59, 61, 62] 
    elements_to_remove = t_rm
    if elements_to_remove:
        subjects = [item for item in subjects if item not in elements_to_remove]
        print("Filtered subjects:", subjects)
    else:
        print("No elements removed, elements_to_remove is empty.")

    val_subjects = subjects[:val_size]
    test_subjects = subjects[val_size:val_size + test_size]
    train_subjects = subjects[val_size + test_size:]
    
    elements_to_add = t_rm
    
    # Add subject back to the training set
    train_subjects.extend(elements_to_add)
    
    np.random.shuffle(train_subjects)

    return train_subjects, val_subjects, test_subjects

# NEW: pass top_percentage explicitly (e.g., 0.2 for top 20%)
def _process_label_class(
    data1, data2, labels,
    window_size, stride, comp_gradient, top_percentage,
    agg_data_1, agg_data_2, agg_labels
):
    if comp_gradient:
        # gradient computed on stream-1; adjust if you prefer stream-2 or a combo
        grad_1 = compute_gradient(data1)

        # gradient windows have length window_size-1 if gradient is a simple diff
        grad_windows_1, _ = sliding_window(
            grad_1, labels, window_size - 1, grad_1.shape[0], window_size, stride
        )
        scores = score_windows_by_gradient(grad_windows_1)
        sorted_scores, sorted_indices = sort_window_scores(scores)  # noqa: F841 (if unused)
        top_count = max(1, int(len(sorted_indices) * top_percentage))
        top_indices = sorted_indices[:top_count]

        data_windows_1, label_windows = sliding_window(
            data1, labels, window_size - 1, data1.shape[0], window_size, stride
        )
        data_windows_2, _ = sliding_window(
            data2, labels, window_size - 1, data2.shape[0], window_size, stride
        )

        agg_data_1.append(data_windows_1[top_indices])
        agg_data_2.append(data_windows_2[top_indices])
        agg_labels.append(label_windows[top_indices, -1])

    else:
        data_windows_1, label_windows = sliding_window(
            data1, labels, window_size - 1, data1.shape[0], window_size, stride
        )
        data_windows_2, _ = sliding_window(
            data2, labels, window_size - 1, data2.shape[0], window_size, stride
        )

        agg_data_1.append(data_windows_1)
        agg_data_2.append(data_windows_2)
        agg_labels.append(label_windows[:, -1])

    return agg_data_1, agg_data_2, agg_labels

def low_pass_filter(data, alpha=0.7): # alpha=0.7-0.85-0.9
    """
    Apply low-pass filter to extract gravity component.
    Args:
        data: numpy array of shape (N, 3) → raw accelerometer
        alpha: smoothing factor (0.8–0.98 typically)
    Returns:
        gravity: estimated gravity component, shape (N, 3)
    """
    gravity = np.zeros_like(data)
    # gravity[0] = data[0]   # initialize with first sample
    # Use first 5-10 samples to estimate initial gravity
    gravity[0] = np.mean(data[:10], axis=0)
    for i in range(1, len(data)):
        gravity[i] = alpha * gravity[i-1] + (1 - alpha) * data[i]
    return gravity

def process_subject_files(
    base_df, subjects, folder1, folder2, window_size, stride, comp_gradient, top_percentage=0.2
):
    """
    Per-file windowing:
      For each subject -> for each file:
        read two streams, align lengths, build per-sample labels, create windows,
        optionally select top windows by gradient, and append immediately.
    """
    all_data_windows_1, all_data_windows_2, all_label_windows = [], [], []

    for subject in subjects:
        subject_files = base_df[base_df['Subject'] == subject]['Filename'].tolist()

        for filename in subject_files:
            path1 = os.path.join(folder1, filename)
            path2 = os.path.join(folder2, filename)

            # activity, trial still parsed from filename; ensure it matches your regex logic
            subj, activity, trial = parse_filename(filename)

            df1 = process_csv(path1)
            df2 = process_csv(path2)

            if df1 is None or df2 is None or df1.empty or df2.empty:
                print(f"Skipping {filename} due to missing or invalid data.")
                continue

            # streams as (N, 3)
            data_array_1 = df1[['x', 'y', 'z']].values
            data_array_2 = df2[['x', 'y', 'z']].values
            
            # --- NEW: extract motion only (linear accel) ---
            # gravity = low_pass_filter(data_array_1)
            # linear_accel = data_array_1 - gravity   # motion only

            # # replace raw accel with motion-only
            # data_array_1 = linear_accel

            # align lengths
            min_len = min(data_array_1.shape[0], data_array_2.shape[0])
            if min_len <= 0:
                print(f"Skipping {filename}: non-positive length after alignment.")
                continue
            data_array_1 = data_array_1[:min_len]
            data_array_2 = data_array_2[:min_len]

            # label for the entire file (per-sample vector)
            label = label_activity(activity)  # 1 = fall, 0 = no-fall/ADL
            labels_vec = np.full((min_len,), int(label), dtype=int)

            # Create windows for THIS file and append immediately
            all_data_windows_1, all_data_windows_2, all_label_windows = _process_label_class(
                data_array_1, data_array_2, labels_vec,
                window_size, stride, comp_gradient, top_percentage,
                all_data_windows_1, all_data_windows_2, all_label_windows
            )

    # Concatenate across all files in this split
    return (
        np.concatenate(all_data_windows_1, axis=0) if all_data_windows_1 else np.array([]),
        np.concatenate(all_data_windows_2, axis=0) if all_data_windows_2 else np.array([]),
        np.concatenate(all_label_windows, axis=0) if all_label_windows else np.array([])
    )

def main_run(input_folder_1, input_folder_2, window_size, stride, val_size, test_size, 
             train_grad, val_grad, test_grad):
    
    print("===== main_run configuration =====")
    print("input_folder_1:", input_folder_1)
    print("input_folder_2:", input_folder_2)
    print("window_size:", window_size)
    print("stride:", stride)
    print("val_size:", val_size)
    print("test_size:", test_size)
    print("train_grad:", train_grad)
    print("val_grad:", val_grad)
    print("test_grad:", test_grad)
    print("==================================")
    
    base_df = parse_csv_filenames(input_folder_1)
    train_subjects, val_subjects, test_subjects = split_subjects(base_df, val_size, test_size)

    print(f"Train subjects: {train_subjects}")
    print(f"Validation subjects: {val_subjects}")
    print(f"Test subjects: {test_subjects}")

    train_data_1, train_data_2, train_labels = process_subject_files(base_df, train_subjects, 
                                                                      input_folder_1, input_folder_2, 
                                                                      window_size, stride, train_grad)
    val_data_1, val_data_2, val_labels = process_subject_files(base_df, val_subjects, 
                                                                input_folder_1, input_folder_2, 
                                                                window_size, stride, val_grad)
    test_data_1, test_data_2, test_labels = process_subject_files(base_df, test_subjects, 
                                                                   input_folder_1, input_folder_2, 
                                                                   window_size, stride, test_grad)

    train_labels = train_labels.reshape(-1, 1)
    val_labels = val_labels.reshape(-1, 1)
    test_labels = test_labels.reshape(-1, 1)    

    print(f"Train data shapes: Stream1 {train_data_1.shape}, Stream2 {train_data_2.shape}, Train labels shape: {train_labels.shape}")
    print(f"Validation data shapes: Stream1 {val_data_1.shape}, Stream2 {val_data_2.shape}, Validation labels shape: {val_labels.shape}")
    print(f"Test data shapes: Stream1 {test_data_1.shape}, Stream2 {test_data_2.shape}, Test labels shape: {test_labels.shape}")


    return train_data_1, train_data_2, train_labels, val_data_1, val_data_2, val_labels, test_data_1, test_data_2, test_labels

#%% Augmentation function
#--------------------------- Augmentation
def jitter(data):
        """
        Add Gaussian noise (jitter).
        Uses fixed sigma = 0.02
        """
        sigma = 0.02
        noise = np.random.normal(loc=0, scale=sigma, size=data.shape)
        return data + noise


def random_crop(data):
        """
        Randomly crop a fraction of the sequence and pad back.
        Uses fixed crop_ratio = 0.9
        """
        crop_ratio = 0.9
        window_size, channels = data.shape
        crop_len = int(window_size * crop_ratio)

        start = np.random.randint(0, window_size - crop_len + 1)
        cropped = data[start:start + crop_len]

        # Pad back to original length
        if cropped.shape[0] < window_size:
            pad_len = window_size - cropped.shape[0]
            cropped = np.pad(cropped, ((0, pad_len), (0, 0)), mode="edge")

        return cropped


def scaling(data):
        """
        Randomly scale the signal amplitude.
        Uses fixed min/max scaling factors (0.8 – 1.2).
        """
        scale_factor = np.random.uniform(0.8, 1.2)
        return data * scale_factor
    
    # -----------------------------
    # New augmentations
    # -----------------------------
def time_warp(data, sigma=0.2, knot=4):
        """
        Apply smooth time warping using cubic spline.
        Args:
            data: (window_size, channels)
            sigma: how strong the warping is
            knot: number of control points for spline
        """
        window_size, channels = data.shape
        orig_steps = np.arange(window_size)

        # Generate random smooth curve
        random_warp = np.random.normal(loc=1.0, scale=sigma, size=knot+2)
        warp_steps = np.linspace(0, window_size-1, num=knot+2)
        spline = CubicSpline(warp_steps, random_warp)
        tt = spline(orig_steps)

        # Cumulative sum to ensure monotonic increasing
        tt_cum = np.cumsum(tt)
        tt_cum = (tt_cum / tt_cum[-1]) * (window_size - 1)

        # Interpolate along the new time axis
        warped = np.zeros_like(data)
        for dim in range(channels):
            warped[:, dim] = np.interp(orig_steps, tt_cum, data[:, dim])

        return warped

def permutation(data, n_segments=4):
        """
        Randomly permute segments of the sequence.
        Args:
            data: (window_size, channels)
            n_segments: number of segments to split into
        """
        window_size, channels = data.shape
        segment_size = window_size // n_segments

        # Split into segments
        segments = [data[i*segment_size:(i+1)*segment_size] for i in range(n_segments)]

        # Shuffle order
        np.random.shuffle(segments)

        # Concatenate back
        return np.concatenate(segments, axis=0)
    
def rotate(data):
        """
        Apply a random 3D rotation to (timesteps, 3) data.
        Rotation is applied only to the (x, y, z) channels.
        """
        # Random small rotation angles (in radians)
        angle_x = np.random.uniform(-np.pi/18, np.pi/18)  # ±10°
        angle_y = np.random.uniform(-np.pi/18, np.pi/18)
        angle_z = np.random.uniform(-np.pi/18, np.pi/18)

        # Rotation matrices
        Rx = np.array([[1, 0, 0],
                    [0, np.cos(angle_x), -np.sin(angle_x)],
                    [0, np.sin(angle_x),  np.cos(angle_x)]])
        
        Ry = np.array([[ np.cos(angle_y), 0, np.sin(angle_y)],
                    [0, 1, 0],
                    [-np.sin(angle_y), 0, np.cos(angle_y)]])
        
        Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                    [np.sin(angle_z),  np.cos(angle_z), 0],
                    [0, 0, 1]])

        # Combined rotation
        R = Rz @ Ry @ Rx

        # Apply rotation to each timestep
        return data @ R.T
#%% Call Augmentation
def augment_data(
        data,
        labels,
        apply_jitter=False, #<--
        apply_crop=True, #<--
        apply_scaling=False,
        apply_timewarp=False, #<--
        apply_permutation=False,
        apply_rotation=False
    ):
    augmented_data = []
    augmented_labels = []
    
    # Check if no augmentation is selected
    if not any([apply_jitter, apply_crop, apply_scaling, apply_timewarp, apply_permutation, apply_rotation]):
        print("No augmentation applied — returning empty arrays.")
        return np.array([]), np.array([])

    # Print what is being applied
    if apply_jitter: print("jitter applied")
    if apply_crop: print("crop applied")
    if apply_scaling: print("scaling applied")
    if apply_timewarp: print("timewarping applied")
    if apply_permutation: print("permutation applied")
    if apply_rotation: print("rotation applied")

    # Apply selected augmentations
    for sample, label in zip(data, labels):
        new_sample = sample.copy()
        if apply_jitter:
            new_sample = jitter(new_sample)
        if apply_crop:
            new_sample = random_crop(new_sample)
        if apply_scaling:
            new_sample = scaling(new_sample)
        if apply_timewarp:
            new_sample = time_warp(new_sample)
        if apply_permutation:
            new_sample = permutation(new_sample)
        if apply_rotation:
            new_sample = rotate(new_sample)

        augmented_data.append(new_sample)
        augmented_labels.append(label)  # keep same label

    return np.array(augmented_data), np.array(augmented_labels)


def augment_training_data(train_data_1, train_data_2, train_labels, mode="falls"):
    """
    mode = "falls" -> augment only fall samples (label=1)
    mode = "all"   -> augment every sample
    """

    # Identify indices
    fall_idx = np.where(train_labels.flatten() == 1)[0]
    nofall_idx = np.where(train_labels.flatten() == 0)[0]

    # Separate data
    fall_data_1, fall_data_2, fall_labels = train_data_1[fall_idx], train_data_2[fall_idx], train_labels[fall_idx]
    nofall_data_1, nofall_data_2, nofall_labels = train_data_1[nofall_idx], train_data_2[nofall_idx], train_labels[nofall_idx]

    if mode == "falls":
        # Augment only fall samples
        aug_fall_data_1, aug_fall_labels = augment_data(fall_data_1, fall_labels)
        aug_fall_data_2, _ = augment_data(fall_data_2, fall_labels)

        train_data_1 = np.concatenate([nofall_data_1, fall_data_1, aug_fall_data_1], axis=0)
        train_data_2 = np.concatenate([nofall_data_2, fall_data_2, aug_fall_data_2], axis=0)
        train_labels = np.concatenate([nofall_labels, fall_labels, aug_fall_labels], axis=0)

    elif mode == "all":
        # Augment everything
        aug_data_1, aug_labels = augment_data(train_data_1, train_labels)
        aug_data_2, _ = augment_data(train_data_2, train_labels)

        train_data_1 = np.concatenate([train_data_1, aug_data_1], axis=0)
        train_data_2 = np.concatenate([train_data_2, aug_data_2], axis=0)
        train_labels = np.concatenate([train_labels, aug_labels], axis=0)

    else:
        raise ValueError("Invalid mode. Choose 'falls' or 'all'.")

    return train_data_1, train_data_2, train_labels
