# Function to compare detected and true annotations
import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from WTdelineator import signalDelineation
from ECGprocessing import bandpass_filter
import logging


def calculate_metrics(true_annotations, detected_annotations, tolerance=5):
    """
    Calculate sensitivity and positive predictivity for a single sample.
    
    Parameters:
        true_annotations (list): Ground truth annotations.
        detected_annotations (list): Detected annotations from the algorithm.
        tolerance (int): Allowed tolerance in samples for matching.
    
    Returns:
        (int, int, int): Counts of TP (True Positives), FP (False Positives), FN (False Negatives).
    """
    TP = 0  # True positives
    FP = 0  # False positives
    FN = 0  # False negatives

    # Create boolean masks for matched annotations
    matched_true = [False] * len(true_annotations)
    matched_detected = [False] * len(detected_annotations)
    
    # Match detected annotations to true annotations
    for i, detected in enumerate(detected_annotations):
        for j, true in enumerate(true_annotations):
            if abs(detected - true) <= tolerance and not matched_true[j]:
                TP += 1
                matched_true[j] = True
                matched_detected[i] = True
                break
    
    # Count unmatched detections as false positives
    FP = sum(not matched for matched in matched_detected)
    
    # Count unmatched true annotations as false negatives
    FN = sum(not matched for matched in matched_true)
    
    return TP, FP, FN



# Function to compute Sensitivity (Se) and Positive Predictivity (+P)
def compute_metrics(detected_qrs, true_qrs, fs, tolerance=0.1):
    tolerance_samples = int(tolerance * fs)  # Convert tolerance to samples
    detected_qrs = np.sort(detected_qrs)
    true_qrs = np.sort(true_qrs)

    TP = 0
    FP = 0
    FN = 0

    unmatched_true_qrs = true_qrs.copy()
    for detected in detected_qrs:
        differences = np.abs(unmatched_true_qrs - detected)
        min_diff = np.min(differences) if len(differences) > 0 else np.inf

        if min_diff <= tolerance_samples:
            TP += 1
            unmatched_true_qrs = unmatched_true_qrs[differences > tolerance_samples]
        else:
            FP += 1

    FN = len(unmatched_true_qrs)
    Se = TP / (TP + FN) if (TP + FN) > 0 else 0
    Pp = TP / (TP + FP) if (TP + FP) > 0 else 0

    return Se, Pp

def check_and_correct_polarity(signal):
    """
    Check the polarity of an ECG signal and correct it if inverted.
    Parameters:
        signal (numpy array): ECG signal.
    Returns:
        corrected_signal (numpy array): Signal with corrected polarity.
    """
    if np.abs(np.min(signal)) > np.abs(np.max(signal)):  # Check if negative deflection is stronger
        return -signal  # Invert polarity
    return signal


def adjust_detected_r_peaks(signal, detected_qrs, window_size=10):
    """
    Adjust detected QRS complexes to align with the maximum amplitude.
    Parameters:
        signal (numpy array): ECG signal.
        detected_qrs (numpy array): Detected QRS indices.
        window_size (int): Number of samples around each detected QRS to search for the maximum.
    Returns:
        adjusted_qrs (numpy array): Adjusted QRS indices aligned with the peaks.
    """
    adjusted_qrs = []
    for r in detected_qrs:
        start = max(0, r - window_size)
        end = min(len(signal), r + window_size)
        max_idx = np.argmax(np.abs(signal[start:end])) + start
        adjusted_qrs.append(max_idx)
    return np.array(adjusted_qrs)


# Process all records in the dataset
def process_records(dataset_dir, tolerance=0.1, low_threshold=0.5):
    records = [f.split('.')[0] for f in os.listdir(dataset_dir) if f.endswith('.dat')]
    records = sorted(set(records))
    
    total_sensitivity = []
    total_predictivity = []
    problematic_records = []

    for record in records:
        try:
            logging.info(f"Processing Record: {record}")
            record_path = f"{dataset_dir}/{record}"

            # Load signal and annotations
            signal, fields = wfdb.rdsamp(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
            fs = fields['fs']
            signal_lead = signal[:, 0]

            # Preprocess signal
            
            filtered_signal = bandpass_filter(signal_lead, fs)
            filtered_signal = check_and_correct_polarity(filtered_signal) 

            # Detect QRS complexes
            _, QRS, _ = signalDelineation(filtered_signal, fs)
            detected_qrs = adjust_detected_r_peaks(filtered_signal, QRS[:, 2], window_size=10)
            # detected_qrs = QRS[:, 2]

            # True QRS annotations
            true_qrs = annotation.sample

            # Compute metrics
            Se, Pp = compute_metrics(detected_qrs, true_qrs, fs, tolerance)
            logging.info(f"Record {record}: Sensitivity (Se): {Se:.2f}, Positive Predictivity (+P): {Pp:.2f}")

            total_sensitivity.append(Se)
            total_predictivity.append(Pp)

            # Diagnose low metrics
            if Se < low_threshold or Pp < low_threshold:
                problematic_records.append(record)
                logging.warning(f"Low metrics for Record {record}: Se={Se:.2f}, +P={Pp:.2f}")
                visualize_record(filtered_signal, fs, detected_qrs, true_qrs, record)

        except Exception as e:
            logging.error(f"Error processing record {record}: {e}")

    # Compute overall metrics
    avg_sensitivity = np.mean(total_sensitivity) if total_sensitivity else 0
    avg_predictivity = np.mean(total_predictivity) if total_predictivity else 0

    logging.info(f"Overall Sensitivity (Se): {avg_sensitivity:.2f}")
    logging.info(f"Overall Positive Predictivity (+P): {avg_predictivity:.2f}")
    logging.info(f"Problematic Records: {problematic_records}")