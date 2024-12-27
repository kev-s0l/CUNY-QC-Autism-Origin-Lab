import mne
import numpy as np
import matplotlib.pyplot as plt

# Load file
edf_file_path = "C:/Users/kevin/Desktop/RESEARCH LAB/DummyData.edf"
raw = mne.io.read_raw_edf(edf_file_path, preload=True)

# Apply a filter to the data
# Low and High Frequency can be adjusted, but best if left as None
raw.filter(l_freq=None, h_freq=None, fir_design="firwin")

# Define frequency bands for sleep stage analysis
frequency_bands = {
    "Delta": (0.5, 3.0),  # Delta waves
    "Theta": (4, 8),      # Theta waves
    "Alpha": (8, 13),     # Alpha waves
    "Beta": (14, 32),     # Beta waves
}

# Function to compute the power in a specific frequency band
def compute_band_power(data, sfreq, band):
    psd, freqs = mne.time_frequency.psd_array_welch(data, sfreq=sfreq, fmin=band[0], fmax=band[1], n_per_seg=256)
    return np.sum(psd)  # Total power in the band

# Select relevant EEG channels for analysis
eeg_channels = [
    "E1:M2",
    "E2:M2",  # EOG (Eye movements)
    "F4:M1",
    "F3:M2",
    "C4:M1",
    "C3:M2",
    "O2:M1",
    "O1:M2",  # EEG (Brain waves)
    "EMG1",
    "EMG2",
    "EMG3",    # EMG (Muscle activity)
    "ECG II",  # ECG (Heart activity)
]
raw.pick_channels(eeg_channels)

# Get the EEG data and sampling frequency
data = raw.get_data(return_times=False)  # Get data only
sfreq = raw.info["sfreq"]  # Get the sampling frequency

# Initialize variables for sleep stage classification
time_segments = 30  # Segment length in seconds
num_segments = int(raw.times[-1] // time_segments)
sleep_stages = np.zeros(num_segments)  # Array to store sleep stages for each segment

# Loop through each time segment to compute band powers
for segment in range(num_segments):
    segment_start = segment * time_segments * sfreq
    segment_end = segment_start + time_segments * sfreq

    if segment_end > data.shape[1]:
        break  # Avoid going out of bounds

    # Compute band powers for each EEG channel in this segment
    band_powers = {band: [] for band in frequency_bands}
    for ch_index, ch_data in enumerate(data):
        ch_segment = ch_data[int(segment_start):int(segment_end)]
        for band, freq_range in frequency_bands.items():
            band_power = compute_band_power(ch_segment, sfreq, freq_range)
            band_powers[band].append(band_power)

    # Classify sleep stage based on defined thresholds
    delta_power = np.mean(band_powers["Delta"])
    theta_power = np.mean(band_powers["Theta"])
    alpha_power = np.mean(band_powers["Alpha"])

    # Example classification logic
    if delta_power > theta_power and delta_power > alpha_power:
        sleep_stages[segment] = 4  # N3 (Deep Sleep)
    elif theta_power > delta_power:
        sleep_stages[segment] = 3  # N2 (Light Sleep)
    elif alpha_power > delta_power and alpha_power > theta_power:
        sleep_stages[segment] = 2  # N1 (Light Sleep)
    else:
        sleep_stages[segment] = 1  # Wake or unclear stage

# Create a time axis for plotting
time_axis = np.arange(num_segments) * time_segments

# Plotting sleep stages
plt.figure(figsize=(15, 6))
plt.step(time_axis, sleep_stages, where="post")
plt.yticks([1, 2, 3, 4], ["Wake", "N1", "N2", "N3"])
plt.xlabel("Time (seconds)")
plt.ylabel("Sleep Stages")
plt.title("Sleep Stages Over Time")
plt.grid()
plt.xlim(0, time_axis[-1])
plt.ylim(0.5, 4.5)
plt.show()
