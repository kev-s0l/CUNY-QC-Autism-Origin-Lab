import mne
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Temporary Solution -- Still a major unkown 
warnings.filterwarnings("ignore")

# Load the EDF file
edf_file_path = "C:/Users/kevin/Desktop/RESEARCH LAB/DummyData.edf"
raw = mne.io.read_raw_edf(edf_file_path, preload=True)

# Apply a filter to the data
# Readings go above built-in threshold --> best left both l_freq & h_freq = None
raw.filter(l_freq=None, h_freq=None, fir_design="firwin")

# Frequency bands catered more towards infants
frequency_bands = {
    "Delta": (0.5, 3.0),  # Delta waves
    "Theta": (4, 8),      # Theta waves
}

# Function to compute the power in a specific frequency band
def compute_band_power(data, sfreq, band):
    psd, freqs = mne.time_frequency.psd_array_welch(
        data, sfreq=sfreq, fmin=band[0], fmax=band[1], n_per_seg=256
    )
    return np.sum(psd)  # Total power in the band

# Select relevant EEG, EOG, EMG, and ECG channels for analysis
selected_channels = [
    "E1", "E2",  # EOG (Eye movements)
    "F4", "F3", "C4", "C3", "O2", "O1",  # EEG (Brain waves)
    "EMG1", "EMG2", "EMG3",  # EMG (Muscle activity)
    "ECG II"  # ECG (Heart activity)
]
raw.pick_channels(selected_channels)

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
    band_powers = {band: [] for band in ["Delta", "Theta"]}
    for ch_index, ch_data in enumerate(data):
        ch_segment = ch_data[int(segment_start):int(segment_end)]
        for band, freq_range in frequency_bands.items():
            band_power = compute_band_power(ch_segment, sfreq, freq_range)
            band_powers[band].append(band_power)

    # Compute average band power across all channels for this segment
    delta_power = np.mean(band_powers["Delta"])
    theta_power = np.mean(band_powers["Theta"])

    # Infant sleep stage classification logic:
    if delta_power > theta_power:  # Quiet Sleep
        sleep_stages[segment] = 4  # N3 (Deep Sleep)
    elif theta_power > delta_power:  # Active Sleep
        sleep_stages[segment] = 5  # REM (Active Sleep)
    elif delta_power == 0 and theta_power == 0:  # No significant power
        sleep_stages[segment] = 1  # Wake
    else:  # Transitional or unclear stage
        sleep_stages[segment] = 2  # N1 (Drowsy or Transitional Sleep)

# Create a time axis for plotting
time_axis = np.arange(num_segments) * time_segments

# Plotting sleep stages
plt.figure(figsize=(15, 6))
plt.step(time_axis, sleep_stages, where="post")
plt.yticks([1, 2, 4, 5], ["Wake", "N1", "N3", "REM"])
plt.xlabel("Time (seconds)")
plt.ylabel("Sleep Stages")
plt.title("Sleep Stages Over Time (Infants)")
plt.grid()
plt.xlim(0, time_axis[-1])
plt.ylim(0.5, 5.5)
plt.show()
