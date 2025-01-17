import matplotlib.pyplot as plt
import WTdelineator as wav
import wfdb
import numpy as np
import os

# Dataset Path and Record Name
dbase = 'staff_data/data'
rec = '042c'


sNum = 1

# Read Signal Data
try:
    s, att = wfdb.rdsamp(f"{dbase}/{rec}")
    print("Signal loaded successfully.")
except FileNotFoundError as e:
    print(f"Signal loading error: {e}")
    exit()

# Manually Handle `.event` File
try:
    # Assuming `.event` is a text file, read it as a fallback
    event_file = os.path.join(dbase, f"{rec}.event")
    with open(event_file, "r") as ef:
        events = ef.readlines()
    print("Event file loaded successfully.")
    # Process `events` if necessary
except FileNotFoundError as e:
    print(f"Annotation loading error: {e}")
    events = []


# Check available signals and select a valid signal index
print(f"Available signals: {att['sig_name']}")
print(f"Number of signals: {len(att['sig_name'])}")

if sNum >= len(att['sig_name']):
    print(f"Invalid signal number {sNum}, defaulting to the first signal.")
    sNum = 0

# Inspect attributes to identify signal name key
print(f"Attributes in `att`: {att}")

# Handle signal name dynamically
if 'sig_name' in att:
    sName = att['sig_name'][sNum]
elif 'sig_name' in att:
    sName = att['sig_name'][sNum]
else:
    print("Signal name not found in attributes. Using default name.")
    sName = f"Signal {sNum}"

print(f"Selected signal: {sName}")

# Signal Metadata
fs = att['fs']
# sName = att.get('signame', ['ECG'])[sNum]

# Range for Signal Analysis
beg = int(np.floor(2 ** 16))
end = int(np.floor(2 * 2 ** 16))
sig = s[beg:end, sNum]
N = sig.shape[0]
t = np.arange(0, N / fs, 1 / fs)

# Wavelet Transform Delineation
Pwav, QRS, Twav = wav.signalDelineation(sig, fs)

# Calculate Biomarkers
QRSd = QRS[:, -1] - QRS[:, 0]
Tind = np.nonzero(Twav[:, 0])
QT = Twav[Tind, -1] - QRS[Tind, 0]
Td = Twav[Tind, -1] - Twav[Tind, 0]
Pind = np.nonzero(Pwav[:, 0])
Pd = Pwav[Pind, -1] - Pwav[Pind, 0]

# Plot Delineation Results
plt.figure()
plt.plot(t, sig, label=sName)
plt.plot(t[QRS[:, 0]], sig[QRS[:, 0]], '*r', label='QRSon', markersize=15)
plt.plot(t[QRS[:, 1]], sig[QRS[:, 1]], '*y', label='Q', markersize=15)
plt.plot(t[QRS[:, 2]], sig[QRS[:, 2]], '*k', label='R', markersize=15)
plt.plot(t[QRS[:, 3]], sig[QRS[:, 3]], '*m', label='S', markersize=15)
plt.plot(t[QRS[:, 4]], sig[QRS[:, 4]], '*g', label='QRSend', markersize=15)
plt.plot(t[Twav[:, 0]], sig[Twav[:, 0]], '^r', label='Ton', markersize=10)
plt.plot(t[Twav[:, 1]], sig[Twav[:, 1]], '^k', label='T1', markersize=10)
plt.plot(t[Twav[:, 2]], sig[Twav[:, 2]], '^m', label='T2', markersize=10)
plt.plot(t[Twav[:, 3]], sig[Twav[:, 3]], '^g', label='Tend', markersize=10)
plt.title('Delineator output')
plt.xlabel('Time (s)')
plt.ylabel('ECG (mV)')
plt.legend()
plt.show()
