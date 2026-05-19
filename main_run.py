
#---------------------------------------------------------
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

from sklearn.metrics import precision_recall_fscore_support, classification_report, roc_auc_score, roc_curve, f1_score, confusion_matrix, auc, roc_auc_score
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy import signal

import pywt
import cv2
import pickle
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MaxPooling1D, LSTM, add, BatchNormalization
from sklearn.utils.class_weight import compute_class_weight

import gc

np.random.seed(42)

#%%
from helping_functions import *
from gated_cnn_model import *

#%% define paths as input variables
checkpoint_dir = 'models/'

input_folder_1 = "Datasets/SmartFallMM-Dataset/young/accelerometer/watch"
input_folder_2 = "Datasets/SmartFallMM-Dataset/young/gyroscope/watch"


#---------------------------------------------------------
top_percentage = 0.8

syndb_name = 'smm_mag_fps_v2'  # Change this based on the model being used

#%% Load datasets and run the models
# Loop through 5 folds
for fold in range(1, 11):  # Starts at 1 and ends at 5 (inclusive of 1, exclusive of 6)
    train_data, train_labels, val_data, val_labels, test_data, test_labels = [], [], [], [], [], []
    
    # Update model title with the current fold
    model_title = f"mmba_cnn_{syndb_name}_fold_{fold}"
    model_name = model_title + '.keras'

    # Construct the checkpoint filepath
    checkpoint_filepath = os.path.join(checkpoint_dir, model_name)

    # Print or use the checkpoint filepath as needed
    print(f"Fold {fold}: {checkpoint_filepath}")
    
    train_data_1, train_data_2, train_labels, val_data_1, val_data_2, val_labels, test_data_1, test_data_2, test_labels = main_run(
        input_folder_1,
        input_folder_2,
        window_size=128,
        stride=10,
        val_size=1,
        test_size=1,
        train_grad=False,
        val_grad=False,
        test_grad=False
    )
    
    # Check the distribution of classes in the training labels
    print("\nBefore Augmentation and adding FP:")
    unique, counts = np.unique(train_labels, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    print(f"Class distribution in training labels: {class_distribution}\n")
    
    unique, counts = np.unique(val_labels, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    print(f"Class distribution in validation labels: {class_distribution}\n")
    
    unique, counts = np.unique(test_labels, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    print(f"Class distribution in test labels: {class_distribution}\n")
    
    # --- Add Magnitude ---
    
    print("Adding features channel...")
    
    train_data_1 = add_features(train_data_1)
    val_data_1 = add_features(val_data_1)
    test_data_1 = add_features(test_data_1)  
    
    train_data_2 = add_features(train_data_2)
    val_data_2 = add_features(val_data_2)
    test_data_2 = add_features(test_data_2)  
    
    print(f"New train_data_1 shape: {train_data_1.shape}")
    print(f"New val_data_1 shape: {val_data_1.shape}")
    print(f"New test_data_1 shape: {test_data_1.shape}\n")
    
    print(f"New train_data_2 shape: {train_data_2.shape}")
    print(f"New val_data_2 shape: {val_data_2.shape}")
    print(f"New test_data_2 shape: {test_data_2.shape}")  

    # ----------------------- Class weights
    # Flatten labels
    train_labels_flat = np.ravel(train_labels)

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels_flat), y=train_labels_flat)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

    print(f"Class weights: {class_weight_dict}")
    
    #----------------------- Convert data to tf.float32
    # Convert data to tf.float32 directly
    train_data_1 = train_data_1.astype(np.float32)
    train_data_2 = train_data_2.astype(np.float32)
    train_labels = train_labels.astype(np.int32)

    val_data_1 = val_data_1.astype(np.float32)
    val_data_2 = val_data_2.astype(np.float32)
    val_labels = val_labels.astype(np.int32)

    test_data_1 = test_data_1.astype(np.float32)
    test_data_2 = test_data_2.astype(np.float32)
    test_labels = test_labels.astype(np.int32)
    
    # ---------------- Define and Compile the model
    
    # Instantiate model and optimizer
    model = build_dual_stream_model(input_shape=(128, 4))
    
    
    # Compile the model
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy", Precision(), Recall()],
    )

    # Print the model summary
    model.summary()

    # -------------------------- build the model
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False, 
        monitor='val_loss',
        mode='min', 
        save_best_only=True, 
        verbose=1  
    )

    # Define the EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',  
        mode='min', 
        patience=10,
        verbose=1  
    )

    training_start_time = time.time()
    history = model.fit(
            [train_data_1, train_data_2],
            train_labels,
            epochs=250, 
            shuffle=True,
            batch_size=32,
            verbose=1,
            validation_data=([val_data_1, val_data_2], val_labels), 
            callbacks=[model_checkpoint, early_stopping],
            class_weight=class_weight_dict
    )

    training_end_time = time.time()
    training_duration = training_end_time - training_start_time
    print(f"Total training time: {training_duration:.2f} seconds.")
    
    # ---------------- Save the history

    with open(checkpoint_dir+'/'+model_title+'.pkl', 'wb') as file:
        pickle.dump(history.history, file)
        
    #---------------- Plot training history
    # Plot training vs validation loss
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    plt.title(f'Training and Validation Loss - Fold {fold}', fontsize=13)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save plot
    output_path = f'output_pngs/{model_title}_loss_curve.png'
    plt.savefig(output_path, format='png', dpi=200)
    plt.close()
    print(f"Saved: {output_path}")
        
    # --------------- Evaluate the model
    # 3. Load the model with custom objects
    model = load_model(checkpoint_filepath, custom_objects={"GatedCNN": GatedCNN})
    
    # Get predicted probabilities for the positive class
    y_pred_prob1 = model.predict([test_data_1, test_data_2], verbose=1).squeeze()

    # ROC AUC Score
    roc_auc = roc_auc_score(test_labels, y_pred_prob1)
    print(f"ROC AUC Score: {roc_auc:.4f}")

    # Evaluate over a range of thresholds
    for i in range(5, 100, 5):
        threshold = i / 100
        y_pred_binary = (y_pred_prob1 >= threshold).astype(int)

        print(f"\nThreshold: {threshold:.2f}")
        print(classification_report(test_labels, y_pred_binary))
        print(f"F1 Score: {f1_score(test_labels, y_pred_binary):.4f}")
        
    # --------------- plot ROC AUC
    # Calculate the ROC curve
    
    fpr, tpr, _ = roc_curve(test_labels, y_pred_prob1)
    roc_auc = auc(fpr, tpr)
    print(f'fold_{fold}', 'ROC curve (area = %0.2f)' % roc_auc)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # Save the figure as a PNG file
    output_path = 'output_pngs/'+model_title+'_roc_curve.png' 
    plt.savefig(output_path, format='png', dpi=200) 
    
    
    # -------------- Confusion Matrix
    # Set the threshold to 0.5
    threshold = 0.5
    print(f"Threshold: {threshold}")

    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred_prob1 >= threshold).astype(int)

    # Print classification report
    print(classification_report(test_labels, y_pred_binary))

    # Calculate F1 score
    f1 = f1_score(test_labels, y_pred_binary)
    print(f"fold_{fold} - Updated F1 Score: {f1:.4f}")

    # Calculate confusion matrix
    cm = confusion_matrix(test_labels, y_pred_binary)
    print(f"fold_{fold} - Confusion Matrix: {cm}")

    # Plot confusion matrix with light pastel colors and increased font size
    plt.figure(figsize=(6, 4), dpi=100)  # Increase DPI for higher resolution
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'shrink': .8},
                annot_kws={"size": 16})  # Increase font size to 16
    plt.xlabel('Predicted Label', fontsize=13)
    plt.ylabel('True Label', fontsize=13)

    # Set custom labels for the x and y axes
    plt.xticks(ticks=[0.5, 1.5], labels=["No Fall", "Fall"], fontsize=14)
    plt.yticks(ticks=[0.5, 1.5], labels=["No Fall", "Fall"], fontsize=14)

    # Save the figure as a PNG file
    output_path = 'output_pngs/'+model_title+'_confusion_matrix.png' 
    plt.savefig(output_path, format='png', dpi=200)
    
    print(f"<=============================Training fold_{fold} ended!=============================>")
    
    tf.keras.backend.clear_session()
    gc.collect()
    
print("All folds are done successfully!")



    
    
    
    
