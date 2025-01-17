import mne
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the EDF file
edf_file_path = "C:/Users/kevin/Desktop/RESEARCH LAB/DummyData.edf"
raw = mne.io.read_raw_edf(edf_file_path, preload=True)

# Apply a filter to the data
raw.filter(l_freq=None, h_freq=None, fir_design='firwin')

# Define frequency bands
frequency_bands = {
    'Delta': (0.5, 4),  # Delta waves (prominent in both sleep and awake states)
    'Theta': (4, 8)     # Theta waves (stronger indicator of sleep in infants)
}

# Function to compute power in a specific frequency band
def compute_band_power(data, sfreq, band):
    psd, freqs = mne.time_frequency.psd_array_welch(data, sfreq=sfreq, fmin=band[0], fmax=band[1], n_per_seg=256)
    return np.sum(psd)

# Select relevant EEG channels
eeg_channels = ["F4:M1", "F3:M2", "C4:M1", "C3:M2", "O2:M1", "O1:M2"]
raw.pick_channels(eeg_channels)

# Get the EEG data and sampling frequency
data = raw.get_data(return_times=False)
sfreq = raw.info['sfreq']

# Initialize variables
time_segments = 30  # Segment length in seconds
num_segments = int(raw.times[-1] // time_segments)
awake_asleep = np.zeros(num_segments)  # Array for awake (0) or asleep (1)

# --- Artifact Analysis Section ---
artifact_annotations = []

for segment in range(num_segments):
    segment_start = segment * time_segments * sfreq
    segment_end = segment_start + time_segments * sfreq
    if segment_end > data.shape[1]:
        break  # Avoid out-of-bounds error

    # Segment data for analysis
    segment_data = data[:, int(segment_start):int(segment_end)]

    # High amplitude artifacts (e.g., > 100 µV in EEG channels)
    high_amplitude = np.any(np.abs(segment_data) > 100e-6)

    # Muscle artifact detection using power in high-frequency bands (30-70 Hz)
    muscle_power = np.mean([
        compute_band_power(segment_data[ch], sfreq, band=(30, 70))
        for ch in range(segment_data.shape[0])
    ])
    muscle_artifact = muscle_power > 0.2  # Threshold for muscle artifact

    # Eye blink artifacts based on extreme amplitude changes
    blink_artifact = False
    for ch in range(segment_data.shape[0]):
        diff = np.diff(segment_data[ch])  # First derivative
        if np.any(np.abs(diff) > 50e-6):  # Large changes suggest eye blinks
            blink_artifact = True
            break

    # Log detected artifacts for this segment
    artifact_detected = high_amplitude or muscle_artifact or blink_artifact
    artifact_annotations.append(artifact_detected)

# Convert artifact annotations to a numpy array for easy manipulation
artifact_annotations = np.array(artifact_annotations)

# --- Sleep Analysis Section ---
for segment in range(num_segments):
    segment_start = segment * time_segments * sfreq
    segment_end = segment_start + time_segments * sfreq
    if segment_end > data.shape[1]:
        break  # Avoid out-of-bounds error

    # Compute average power in Delta and Theta bands across channels
    delta_power = np.mean([compute_band_power(data[ch, int(segment_start):int(segment_end)], sfreq, frequency_bands['Delta']) for ch in range(data.shape[0])])
    theta_power = np.mean([compute_band_power(data[ch, int(segment_start):int(segment_end)], sfreq, frequency_bands['Theta']) for ch in range(data.shape[0])])

    # Determine awake or asleep state using delta and theta bands
    if theta_power > delta_power * 0.75:  # Threshold: theta should be dominant or balanced with delta
        awake_asleep[segment] = 1  # Asleep
    else:
        awake_asleep[segment] = 0  # Awake

# Calculate total awake and asleep time
awake_time = np.sum(awake_asleep == 0) * time_segments / 60  # minutes
asleep_time = np.sum(awake_asleep == 1) * time_segments / 60  # minutes

# Create a time axis for plotting
time_axis = np.arange(num_segments) * time_segments

# --- Combined Graph ---
plt.figure(figsize=(15, 8))

# Plot awake/asleep states
plt.step(time_axis, awake_asleep, where='post', color='purple', label='Awake/Asleep States')
plt.yticks([0, 1], ['Awake', 'Asleep'])
plt.ylabel("Awake/Asleep State", color='purple')
plt.ylim(-0.5, 1.5)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Add a second y-axis for artifact detection
ax2 = plt.gca().twinx()
ax2.step(time_axis, artifact_annotations, where='post', color='red', label='Artifacts Detected')
ax2.set_ylabel("Artifact Detection", color='red')
ax2.set_yticks([0, 1])
ax2.set_yticklabels(['No Artifact', 'Artifact'])
ax2.set_ylim(-0.5, 1.5)

# Title and labels
plt.title("Awake vs Asleep States and Artifact Detection Over Time")
plt.xlabel("Time (seconds)")

# Add legends
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# Print total awake and asleep times
print(f"Total Awake Time: {awake_time:.2f} minutes")
print(f"Total Asleep Time: {asleep_time:.2f} minutes")
