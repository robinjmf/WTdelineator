#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 14:01:44 2018

@author: carlos
"""

#%%
import matplotlib.pyplot as plt
import h5py
import wfdb
import numpy as np
import csv
import ECGprocessing as ecg
from evaluation import calculate_metrics
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

dbase = 'staff_data/data'
fName = 'delinResults.h5'
tests = ['BR', 'BC1', 'BC2', 'BI1', 'BI2', 'BI3', 'BI4', 'BI5', 'PC1', 'PC2', 'PR1', 'PR2']
leadNames = ['V1','V2','V3','V4','V5','V6','DI','DII','DIII','aVR','aVL','aVF']
h5attr = [['Annotation names'], ['Pon,P,Pend, QRSon,R,QRSend,Ton,Tp,Tend']]

# Read annotations from the purpose-made file
annot = []
with open('annotations.csv') as csvfile:
    ann = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in ann:
        annot += [row]

# Initialize lists to store aggregate metrics
r_intervals = []
qrs_durations = []
t_wave_amplitudes = []

#%% Open file to save the results
outFile = h5py.File(fName, 'w')

# Process one patient at a time
for patient in range(1, len(annot)):

    # Log the progress message
    progress = ((patient) / (len(annot) - 1)) * 100
    logging.info(f"Processing patient {patient}: Progress: {progress:.2f}% done")

    # Process BR, BC1, and BC2
    for measurement in range(0, 3):
        rec = annot[patient][measurement]  # Find the name of the record
        if rec != '':  # Check that the record exists
            try:
                s, att = wfdb.rdsamp(f'{dbase}/{rec.zfill(4)}')  # Read from local directory
                # Calculate augmented limb leads and append them to the signals
                aVR, aVL, aVF = ecg.augmentedLimbs(s[:, -3], s[:, -2])
                s = np.concatenate((s, aVR, aVL, aVF), axis=1)

                # Delineate all the ECG leads using the WT and fusion techniques
                ECGdelin = ecg.delineateMultiLeadECG(s, att['fs'])

                # Extract metrics
                for idx, lead_result in enumerate(ECGdelin):  # Enumerate to get lead index and result
                    # Calculate R-R intervals
                    r_locs = lead_result[:, 4]  # R peaks
                    r_intervals_lead = np.diff(r_locs) / att['fs'] * 1000  # Convert to ms
                    r_intervals += list(r_intervals_lead)

                    # Calculate QRS durations
                    qrs_durations_lead = (lead_result[:, 5] - lead_result[:, 3]) / att['fs'] * 1000  # QRS end - QRS onset
                    qrs_durations += list(qrs_durations_lead)

                    # Calculate T-wave amplitudes
                    t_amplitudes_lead = s[lead_result[:, 7], idx]  # Access the current lead column using idx
                    t_wave_amplitudes += list(t_amplitudes_lead)

                # Save results to the HDF5 file
                grp = outFile.create_group(tests[measurement] + '/' + str(patient).zfill(3))
                for idx, ECG in enumerate(ECGdelin):
                    dsetName = leadNames[idx]
                    dset = grp.create_dataset(dsetName, ECG.shape, ECG.dtype)
                    dset[...] = ECG

            except ValueError:
                logging.error(f"There was an error when reading file {rec}, file skipped.")

    # Process BI1 - BI5
    k = 3
    for measurement in range(3, 12, 2):
        rec = annot[patient][measurement]  # Find the name of the record
        if rec != '':  # Check that the record exists
            try:
                s, att = wfdb.rdsamp(f'{dbase}/{rec.zfill(4)}')  # Read from local directory

                # Read the annotations
                a1, a2, _ = annot[patient][measurement+1].split(';')
                a1 = int(int(a1)*att['fs'])  # Start of balloon inflation
                a2 = int(int(a2)*att['fs'])  # End of balloon inflation

                # Only keep the ischemic part of the signal
                s = s[a1:a1+a2, :]

                # Calculate augmented limb leads and append them to the signals
                aVR, aVL, aVF = ecg.augmentedLimbs(s[:, -3], s[:, -2])
                s = np.concatenate((s, aVR, aVL, aVF), axis=1)

                # Delineate all the ECG leads using the WT and fusion techniques
                ECGdelin = ecg.delineateMultiLeadECG(s, att['fs'])

                # Extract metrics (similar to above)
                for idx, lead_result in enumerate(ECGdelin):
                    # Calculate R-R intervals
                    r_locs = lead_result[:, 4]  # R peaks
                    r_intervals_lead = np.diff(r_locs) / att['fs'] * 1000  # Convert to ms
                    r_intervals += list(r_intervals_lead)

                    # Calculate QRS durations
                    qrs_durations_lead = (lead_result[:, 5] - lead_result[:, 3]) / att['fs'] * 1000  # QRS end - QRS onset
                    qrs_durations += list(qrs_durations_lead)

                    # Calculate T-wave amplitudes
                    t_amplitudes_lead = s[lead_result[:, 7], idx]  # Access the current lead column using idx
                    t_wave_amplitudes += list(t_amplitudes_lead)

                # Save results to the HDF5 file
                grp = outFile.create_group(tests[k] + '/' + str(patient).zfill(3))
                for idx, ECG in enumerate(ECGdelin):
                    dsetName = leadNames[idx]
                    dset = grp.create_dataset(dsetName, ECG.shape, ECG.dtype)
                    dset[...] = ECG

            except ValueError:
                logging.error(f"There was an error when reading file {rec}, file skipped.")
        k += 1

# Compute aggregate metrics
mean_r_interval = np.mean(r_intervals) if r_intervals else None
mean_qrs_duration = np.mean(qrs_durations) if qrs_durations else None
mean_t_wave_amplitude = np.mean(t_wave_amplitudes) if t_wave_amplitudes else None

# Log aggregate metrics
logging.info(f"Mean R-R Interval: {mean_r_interval:.2f} ms")
logging.info(f"Mean QRS Duration: {mean_qrs_duration:.2f} ms")
logging.info(f"Mean T-Wave Amplitude: {mean_t_wave_amplitude:.2f} mV")

# Close file
outFile.close()



# import os
# import matplotlib.pyplot as plt
# import h5py
# import wfdb
# import numpy as np
# import csv
# import ECGprocessing as ecg
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# dbase = 'mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0'  # Adjust as needed for your datasets
# fName = 'delinResults.h5'
# tests = ['BR', 'BC1', 'BC2', 'BI1', 'BI2', 'BI3', 'BI4', 'BI5', 'PC1', 'PC2', 'PR1', 'PR2']
# leadNames = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'DI', 'DII', 'DIII', 'aVR', 'aVL', 'aVF']
# h5attr = [['Annotation names'], ['Pon,P,Pend, QRSon,R,QRSend,Ton,Tp,Tend']]

# # Initialize metrics
# r_intervals = []
# qrs_durations = []
# t_wave_amplitudes = []

# # Initialize metrics for Sensitivity and Positive Predictivity
# total_tp = 0
# total_fp = 0
# total_fn = 0

# #%% Open file to save the results
# outFile = h5py.File(fName, 'w')

# # Process one patient at a time
# for patient in range(1, len(tests)):  # Adjust based on dataset
#     progress = ((patient) / (len(tests) - 1)) * 100
#     logging.info(f"Processing patient {patient}: Progress: {progress:.2f}% done")

#     for measurement in range(0, 3):  # Modify as needed for the dataset
#         rec = f"{patient:03}"  # Use appropriate record naming convention
#         try:
#             # Load signal and metadata
#             record_path = os.path.join(dbase, rec)
#             s, att = wfdb.rdsamp(record_path)  # Read the signal
            
#             # Check if annotation file exists
#             annotation_file = f"{record_path}.atr"
#             annotations_available = os.path.isfile(annotation_file)
            
#             # If annotations are available, load them
#             if annotations_available:
#                 annotation = wfdb.rdann(record_path, 'atr')  # Load annotations
#                 true_r_peaks = annotation.sample  # Ground truth R-peaks
#             else:
#                 true_r_peaks = None

#             # Calculate augmented limb leads and append them to the signals
#             aVR, aVL, aVF = ecg.augmentedLimbs(s[:, -3], s[:, -2])
#             s = np.concatenate((s, aVR, aVL, aVF), axis=1)

#             # Delineate all the ECG leads using the WT and fusion techniques
#             ECGdelin = ecg.delineateMultiLeadECG(s, att['fs'])

#             # Extract metrics
#             for idx, lead_result in enumerate(ECGdelin):
#                 # Calculate R-R intervals
#                 r_locs = lead_result[:, 4]  # R peaks
#                 r_intervals_lead = np.diff(r_locs) / att['fs'] * 1000  # Convert to ms
#                 r_intervals += list(r_intervals_lead)

#                 # Calculate QRS durations
#                 qrs_durations_lead = (lead_result[:, 5] - lead_result[:, 3]) / att['fs'] * 1000  # QRS end - QRS onset
#                 qrs_durations += list(qrs_durations_lead)

#                 # Calculate T-wave amplitudes
#                 t_amplitudes_lead = s[lead_result[:, 7], idx]  # Access the current lead column using idx
#                 t_wave_amplitudes += list(t_amplitudes_lead)

#                 # If annotations are available, calculate TP, FP, FN
#                 if true_r_peaks is not None:
#                     detected_r_peaks = lead_result[:, 4]
#                     tp = sum([1 for r in detected_r_peaks if any(abs(r - t) <= 10 for t in true_r_peaks)])
#                     fp = len(detected_r_peaks) - tp
#                     fn = len(true_r_peaks) - tp

#                     # Update aggregate metrics
#                     total_tp += tp
#                     total_fp += fp
#                     total_fn += fn

#             # Save results to the HDF5 file
#             grp = outFile.create_group(tests[measurement] + '/' + str(patient).zfill(3))
#             for idx, ECG in enumerate(ECGdelin):
#                 dsetName = leadNames[idx]
#                 dset = grp.create_dataset(dsetName, ECG.shape, ECG.dtype)
#                 dset[...] = ECG

#         except ValueError:
#             logging.error(f"There was an error when reading file {rec}, file skipped.")

# # Compute aggregate metrics
# mean_r_interval = np.mean(r_intervals) if r_intervals else None
# mean_qrs_duration = np.mean(qrs_durations) if qrs_durations else None
# mean_t_wave_amplitude = np.mean(t_wave_amplitudes) if t_wave_amplitudes else None

# # Compute Sensitivity and Positive Predictivity if annotations are available
# if total_tp + total_fn > 0 and total_tp + total_fp > 0:
#     sensitivity = total_tp / (total_tp + total_fn)
#     positive_predictivity = total_tp / (total_tp + total_fp)
# else:
#     sensitivity = None
#     positive_predictivity = None

# # Log final metrics
# logging.info(f"Mean R-R Interval: {mean_r_interval:.2f} ms")
# logging.info(f"Mean QRS Duration: {mean_qrs_duration:.2f} ms")
# logging.info(f"Mean T-Wave Amplitude: {mean_t_wave_amplitude:.2f} mV")

# if sensitivity is not None and positive_predictivity is not None:
#     logging.info(f"Sensitivity (Se): {sensitivity:.2f}")
#     logging.info(f"Positive Predictivity (+P): {positive_predictivity:.2f}")
# else:
#     logging.info("Sensitivity (Se) and Positive Predictivity (+P) cannot be computed as annotations are not available.")

# # Close file
# outFile.close()
