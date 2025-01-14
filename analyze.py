import h5py
import matplotlib.pyplot as plt

with h5py.File('delinResults.h5', 'r') as f:
    print(f.keys())  # List the top-level groups (tests)
    print(f['BR/001'].keys())  # List ECG leads for patient 001 under 'BR' test


    ecg_v2 = f['BR/001']['V2'][:]  # Extract V2 lead for patient 001, BR test
    plt.plot(ecg_v2)
    plt.title('ECG V2 for Patient 001')
    plt.xlabel('Time (s)')
    plt.ylabel('ECG Amplitude (mV)')
    plt.show()
