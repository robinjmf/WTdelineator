import h5py
import matplotlib.pyplot as plt
import wfdb
import os
import random
import struct

# with h5py.File('delinResults.h5', 'r') as f:
#     print(f.keys())  # List the top-level groups (tests)
#     print(f['BR/001'].keys())  # List ECG leads for patient 001 under 'BR' test


#     ecg_v2 = f['BR/001']['V2'][:]  # Extract V2 lead for patient 001, BR test
#     plt.plot(ecg_v2)
#     plt.title('ECG V2 for Patient 001')
#     plt.xlabel('Time (s)')
#     plt.ylabel('ECG Amplitude (mV)')
#     plt.show()


# # # Path to the .atr file (without the extension)
# record_path = 'mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0/101.atr' 
# file_path = 'mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0/101'

# try:
#     annotation = wfdb.rdann(file_path, 'atr')
#     print("Annotation Samples:", annotation.sample)
#     print("Annotation Symbols:", annotation.symbol)
# except Exception as e:
#     print(f"Error parsing record 101: {e}")

#     import struct


# # Open the file and read raw binary content
# with open(record_path, 'rb') as f:
#     raw_bytes = f.read()

# # Initialize parsing variables
# annotations = []
# i = 0
# current_sample = 0

# while i < len(raw_bytes) - 2:
#     # Parse 2-byte sample difference
#     sample_diff = struct.unpack('<H', raw_bytes[i:i + 2])[0]
#     current_sample += sample_diff  # Accumulate the sample index
#     i += 2

#     # Parse 1-byte annotation type
#     annotation_type = raw_bytes[i]
#     i += 1

#     # Stop parsing if the annotation type is the termination marker
#     if annotation_type == 0:
#         break

#     # Append parsed annotation (sample index and type)
#     annotations.append((current_sample, annotation_type))

# # Map annotation types to their symbols (based on WFDB annotation codes)
# annotation_mapping = {
#     78: "N",   # Normal beat
#     5: "B",    # Beat
#     3: "+",    # Rhythm change
#     65: "A",   # Auxiliary
#     76: "L",   # Left bundle branch block
#     88: "R",   # Right bundle branch block
#     106: "V",  # Premature ventricular contraction
#     46: "/",   # Paced beat
#     42: "F"    # Fusion of ventricular and normal beat
# }
# for sample, annotation_type in annotations:
#     symbol = annotation_mapping.get(annotation_type, "?")
#     if symbol == "?":
#         print(f"Unknown annotation type: {annotation_type}")


# # Base directory of the dataset
# data_dir = 'mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0'

# # List all records in the dataset
# records = [f.split('.')[0] for f in os.listdir(data_dir) if f.endswith('.atr')]

# # Mapping for annotation types
# annotation_mapping = {
#     78: "N",   # Normal beat
#     5: "B",    # Beat
#     3: "+",    # Rhythm change
#     65: "A",   # Auxiliary
#     76: "L",   # Left bundle branch block
#     88: "R",   # Right bundle branch block
#     106: "V",  # Premature ventricular contraction
#     # Add other mappings as needed
# }

# # Process each record
# for record in records:
#     file_path = os.path.join(data_dir, record)
#     with open(file_path + '.atr', 'rb') as f:
#         raw_bytes = f.read()
    
#     annotations = []
#     i = 0
#     current_sample = 0

#     while i < len(raw_bytes) - 2:
#         # Parse 2-byte sample difference
#         sample_diff = struct.unpack('<H', raw_bytes[i:i + 2])[0]
#         current_sample += sample_diff  # Accumulate the sample index
#         i += 2

#         # Parse 1-byte annotation type
#         annotation_type = raw_bytes[i]
#         i += 1

#         # Stop parsing if the annotation type is the termination marker
#         if annotation_type == 0:
#             break

#         # Append parsed annotation (sample index and type)
#         annotations.append((current_sample, annotation_type))

#     # Translate annotations
#     translated_annotations = [
#         (sample, annotation_mapping.get(atype, "?")) for sample, atype in annotations
#     ]

#     # Output results for the record
#     print(f"Annotations for record {record}:")
#     print(translated_annotations[:10])  # Print first 10 annotations


# # Translate annotations
# translated_annotations = [
#     (sample, annotation_mapping.get(atype, "?")) for sample, atype in annotations
# ]

# # Print parsed annotations
# print("Parsed Annotations (Sample Index, Symbol):", translated_annotations[:10])  # Show first 10

# # Read the raw bytes of the .atr file
# with open(record_path + '.atr', 'rb') as f:
#     atr_bytes = f.read()

# # Print the first 50 bytes for inspection
# print(atr_bytes[:50])





#--------------------------------------------------------------



# data_dir = 'mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0'

# # List all records in the dataset
# records = [f.split('.')[0] for f in os.listdir(data_dir) if f.endswith('.dat')]
# records = list(set(records))  # Ensure unique record names

# # Randomly select a record
# random_record = random.choice(records)
# print(f"Selected Record: {random_record}")

# try:
#     # Load the ECG signal
#     record, fields = wfdb.rdsamp(os.path.join(data_dir, random_record))  # Unpack the tuple
#     signal = record[:, 0]  # Access the signal array directly (first lead)
#     fs = fields['fs']  # Get the sampling frequency from the metadata

#     # Check if annotation file exists
#     annotation_path = os.path.join(data_dir, f"{random_record}.atr")
#     if not os.path.exists(annotation_path):
#         raise FileNotFoundError(f"Annotation file {annotation_path} not found.")

#     # Load the annotations
#     annotation = wfdb.rdann(os.path.join(data_dir, random_record), 'atr', sampto=300000)


#     # Display annotations
#     print("Annotations:")
#     for i, (sample, symbol, aux) in enumerate(zip(annotation.sample, annotation.symbol, annotation.aux_note)):
#         print(f"Annotation {i+1}: Sample Index = {sample}, Type = {symbol}, Auxiliary = {aux}")

#     # Convert sample indices to time
#     annotation_times = [sample / fs for sample in annotation.sample]

#     # Plot the ECG signal with annotations
#     plt.figure(figsize=(12, 6))
#     plt.plot(signal, label='ECG Signal', color='blue')
#     plt.scatter(annotation.sample, signal[annotation.sample], color='red', label='Annotations', zorder=5)
#     for i, symbol in enumerate(annotation.symbol):
#         plt.text(annotation.sample[i], signal[annotation.sample[i]], symbol, color='green', fontsize=8)
#     plt.title(f'ECG Signal with Annotations for Record {random_record}')
#     plt.xlabel('Sample Index')
#     plt.ylabel('Amplitude (mV)')
#     plt.legend()
#     plt.show()

# except ValueError as e:
#     print(f"Error processing record {random_record}: {e}")
# except FileNotFoundError as e:
#     print(e)
# except Exception as e:
#     print(f"Unexpected error: {e}")


# with open(os.path.join(data_dir, f"{random_record}.atr"), "rb") as f:
#     data = f.read()
# print(data[:100])  # Display the first 100 bytes


#--------------------------------------------------------------

# import pandas as pd

# # Load the annotations.csv file
# annotations_path = 'annotations.csv'
# annotations_df = pd.read_csv(annotations_path)

# # Display the column names and example rows
# print("Columns in the file:", annotations_df.columns)
# print("First few rows of the file:")
# print(annotations_df.head())

# # Example: Parse specific columns
# if 'BI1:D0;D1;D2' in annotations_df.columns:
#     annotations_parsed = annotations_df['BI1:D0;D1;D2'].dropna().apply(
#         lambda x: [int(val) for val in x.split(';')] if isinstance(x, str) else None
#     )
#     print("Parsed Annotations (Start, End, Type):", annotations_parsed.head())


#--------------------------------------------------------------

import h5py

# Path to the HDF5 file
hdf5_file_path = "delinResults.h5"

def inspect_hdf5(file_path):
    """Recursively inspect the structure and content of an HDF5 file."""
    with h5py.File(file_path, "r") as hdf:
        def print_structure(name, obj):
            indent = '  ' * (name.count('/') - 1)
            if isinstance(obj, h5py.Group):
                print(f"{indent}- Group: {name}")
            elif isinstance(obj, h5py.Dataset):
                print(f"{indent}- Dataset: {name} | Shape: {obj.shape} | Type: {obj.dtype}")
        
        print("HDF5 File Structure:")
        hdf.visititems(print_structure)
        
        # Extract and display some sample data
        print("\nSample Data from the File:")
        for test in hdf.keys():
            test_group = hdf[test]
            print(f"Test: {test}")
            for patient in test_group.keys():
                patient_group = test_group[patient]
                print(f"  Patient: {patient}")
                for lead in patient_group.keys():
                    lead_data = patient_group[lead][:]
                    print(f"    Lead: {lead} | Delineation Data Sample:")
                    print(f"      {lead_data[:5]}")  # Display the first 5 rows of data
                    break  # Limit to one lead for brevity
                break  # Limit to one patient for brevity
            break  # Limit to one test for brevity

# Call the function
inspect_hdf5(hdf5_file_path)

