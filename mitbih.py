import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from WTdelineator import signalDelineation
from evaluation import *
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



# Run the analysis
dataset_dir = "mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0"


# Function to visualize ECG signal with QRS annotations
def visualize_record(signal, fs, detected_qrs, true_qrs, record_name):
    t = np.arange(len(signal)) / fs
    plt.figure(figsize=(12, 6))
    plt.plot(t, signal, label="ECG Signal", color="blue")
    plt.scatter(detected_qrs / fs, signal[detected_qrs], color="red", label="Detected QRS")
    plt.scatter(true_qrs / fs, signal[true_qrs], color="green", label="True QRS")
    plt.title(f"Detected vs. True QRS for Record {record_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.legend()
    plt.show()



def visualize_preprocessing(raw_signal, filtered_signal, fs, record_name):
    t = np.arange(len(raw_signal)) / fs
    plt.figure(figsize=(12, 6))
    plt.plot(t, raw_signal, label="Raw Signal", color="blue", alpha=0.7)
    plt.plot(t, filtered_signal, label="Filtered Signal", color="red")
    plt.title(f"Preprocessing for Record {record_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.legend()
    plt.show()

def analyze_problematic_samples(dataset_dir, record_names, tolerance=0.1, low_threshold=0.5):
    problematic_records = []
    for record_name in record_names:
        try:
            # Load signal and annotations
            record_path = f"{dataset_dir}/{record_name}"
            signal, fields = wfdb.rdsamp(record_path)
            annotation = wfdb.rdann(record_path, 'atr')

            raw_signal = signal[:, 0]  # Use the first lead
            fs = fields['fs']

            # Preprocess signal
            filtered_signal = bandpass_filter(raw_signal, fs)
            visualize_preprocessing(raw_signal, filtered_signal, fs, record_name)
            filtered_signal = check_and_correct_polarity(filtered_signal) # Check and correct polarity
            # Detect QRS complexes
            _, QRS, _ = signalDelineation(filtered_signal, fs)
            detected_qrs = adjust_detected_r_peaks(filtered_signal, QRS[:, 2], window_size=10) # Adjust detected R-peaks
            # detected_qrs = QRS[:, 2]
            print(f"Record {record_name} - Detected QRS: {detected_qrs}")

            # True QRS locations
            true_qrs = annotation.sample
            
            # Compute metrics
            Se, Pp = compute_metrics(detected_qrs, true_qrs, fs, tolerance)
            logging.info(f"Record {record_name}: Sensitivity (Se): {Se:.2f}, Positive Predictivity (+P): {Pp:.2f}")

            # Diagnose low metrics
            if Se < low_threshold or Pp < low_threshold:
                problematic_records.append(record_name)
                logging.warning(f"Low metrics for Record {record_name}: Se={Se:.2f}, +P={Pp:.2f}")
                visualize_record(filtered_signal, fs, detected_qrs, true_qrs, record_name)

        except Exception as e:
            logging.error(f"Error processing record {record_name}: {str(e)}")

    logging.info(f"Problematic Records: {problematic_records}")

# record_names = ['102', '104', '107', '108', '109', '111', '114', '118', '121', '122', '124', '203', '217']
# analyze_problematic_samples(dataset_dir = dataset_dir, record_names = record_names)


process_records(dataset_dir)
#--------------------------------------------------------------


# import matplotlib.pyplot as plt
# import numpy as np
# import wfdb
# import logging  # Assumes you have a function for metrics
# from WTdelineator import signalDelineation

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Settings
# dataset_path = "mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0" 
# tolerance = 0.1  # Tolerance in seconds for true-positive detection
# low_sensitivity_threshold = 0.5  # Threshold for triggering diagnostics
# problematic_records = []

# # Diagnostic Functions
# def visualize_record(signal, fs, detected_qrs, true_qrs, record_name):
#     """Visualize the ECG signal with detected and true QRS annotations."""
#     plt.figure(figsize=(12, 6))
#     time = np.arange(len(signal)) / fs
#     plt.plot(time, signal, label="ECG Signal", color="blue")
#     plt.scatter(detected_qrs / fs, signal[detected_qrs], color="red", label="Detected QRS", zorder=5)
#     plt.scatter(true_qrs / fs, signal[true_qrs], color="green", label="True QRS", zorder=5)
#     plt.title(f"ECG Signal and QRS Annotations for Record {record_name}")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Amplitude (mV)")
#     plt.legend()
#     plt.show()

# def log_false_negatives_and_positives(detected_qrs, true_qrs, fs, tolerance):
#     """Log false negatives and false positives."""
#     false_negatives = [qrs for qrs in true_qrs if not np.any(np.abs(detected_qrs - qrs) <= int(tolerance * fs))]
#     false_positives = [qrs for qrs in detected_qrs if not np.any(np.abs(true_qrs - qrs) <= int(tolerance * fs))]

#     logging.warning(f"False Negatives (Missed QRS Annotations): {false_negatives}")
#     logging.warning(f"False Positives (Incorrectly Detected QRS): {false_positives}")
#     return false_negatives, false_positives

# # Main Processing Loop
# overall_tp, overall_fn, overall_fp = 0, 0, 0

# for record_num in range(100, 200):  # Modify range as per available dataset
#     try:
#         record_path = f"{dataset_path}/{record_num:03d}"
#         logging.info(f"Processing Record: {record_num}")
        
#         # Load the ECG signal and annotations
#         record, fields = wfdb.rdsamp(record_path)
#         annotation = wfdb.rdann(record_path, 'atr')
#         signal = record[:, 0]
#         fs = fields['fs']
        
#         # Preprocess signal
#         filtered_signal = bandpass_filter(signal, fs)

#         # Detect QRS using the delineation function
#         _, QRS, _ = signalDelineation(filtered_signal, fs)
#         detected_qrs = QRS[:, 2]  # Extract R-peak indices

#         # True QRS locations
#         true_qrs = annotation.sample

#         # Compute metrics
#         Se, Pp, TP, FN, FP = compute_metrics(detected_qrs, true_qrs, fs, tolerance)
#         overall_tp += TP
#         overall_fn += FN
#         overall_fp += FP

#         logging.info(f"Record {record_num}: Sensitivity (Se): {Se:.2f}, Positive Predictivity (+P): {Pp:.2f}")

#         # Diagnostics for low sensitivity
#         if Se < low_sensitivity_threshold:
#             logging.warning(f"Low metrics for Record {record_num}: Se={Se:.2f}, +P={Pp:.2f}")
#             problematic_records.append(record_num)

#             # Visualize signal and annotations
#             visualize_record(filtered_signal, fs, detected_qrs, true_qrs, record_name=record_num)

#             # Log false negatives and positives
#             log_false_negatives_and_positives(detected_qrs, true_qrs, fs, tolerance)

#     except Exception as e:
#         logging.error(f"Error processing record {record_num}: {e}")

# # Compute overall metrics
# overall_sensitivity = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
# overall_positive_predictivity = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0

# logging.info(f"Overall Sensitivity (Se): {overall_sensitivity:.2f}")
# logging.info(f"Overall Positive Predictivity (+P): {overall_positive_predictivity:.2f}")
# logging.info(f"Problematic Records: {problematic_records}")
