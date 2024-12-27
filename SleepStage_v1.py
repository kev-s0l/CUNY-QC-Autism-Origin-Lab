import mne
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Look into frequency warnings...
warnings.filterwarnings("ignore")

# Change to whatever file will be analyzed
edf_file_path = "C:/Users/kevin/Desktop/RESEARCH LAB/DummyData.edf"
raw = mne.io.read_raw_edf(edf_file_path, preload=True)

# Apply a filter to the data (1 - 180 Hz bandpass filter to keep relevant frequencies)
raw.filter(l_freq=1.0, h_freq=120.0, fir_design='firwin')

# Relevant data signals used for basic sleep analysis
relevant_signals = [
    "E1:M2", "E2:M2",  # EOG (Eye movements)
    "F4:M1", "F3:M2", "C4:M1", "C3:M2", "O2:M1", "O1:M2",  # EEG (Brain waves)
    "EMG1", "EMG2", "EMG3",  # EMG (Muscle activity)
    "ECG II"  # ECG (Heart activity)
]

# Run through all available signal channels 
available_channels = raw.ch_names
print("Available channels:", available_channels)

# Select only the relevant channels out of available channels --> these will be used for analysis
selected_channels = []
for ch in relevant_signals:
    if ch in available_channels:
        selected_channels.append(ch)

# Output to console which channels were found and selected
if selected_channels:
    raw.pick_channels(selected_channels)
    print(f"Selected relevant channels: {selected_channels}")
else:
    print("None of the relevant channels were found in the EDF file.")

# Plot the selected signals
raw.plot(scalings='auto', title='Selected Signals for Sleep Cycle Detection')

# Check for annotations such as "Sleep, N1, etc." inside EDF file
if len(raw.annotations) > 0:
    print("Annotations found:", raw.annotations)
else:
    print("No annotations found in the EDF file.")

# Print annotations to understand the descriptions
for annot in raw.annotations:
    description = annot['description']
    onset = annot['onset']
    print(f"Annotation description: {description}, onset: {onset}")

# Using EDF_InfoOutput.py --> configure each annotation to its respective sleep stage
stage_mapping = {
    'Wake': 1,  # Wake
    'Snore': 2,  # Light Sleep (N1)
    'Deep Breaths In and Out': 3,  # N2
    'Left Foot Movement': 2,  # Light Sleep (N1)
    'Eyes closed': 3,  # N2
    'Breast Breathing': 4  # Deep Sleep (N3)
}

# Convert annotations to events based on sleep stages
events = []
for annot in raw.annotations:
    if annot['description'] in stage_mapping:
        onset_sample = int(annot['onset'] * raw.info['sfreq'])
        event_id = stage_mapping.get(annot['description'], -1)
        events.append([onset_sample, 0, event_id])

# Some error handling --> checks to see if events were made
if len(events) > 0:
    events_array = np.array(events)
    print("Events array created.")
else:
    print("No events were created. Check if annotations match the stage mapping.")

# Define event IDs for plotting sleep stages
event_id = {
    'Wake': 1,
    'N1': 2,
    'N2': 3,
    'N3': 4
}

# Plot events (sleep stages) only if events exist
if len(events) > 0:
    events_exist = True
    sampling_frequency = raw.info['sfreq']
    first_sample = raw.first_samp
    mne.viz.plot_events(events_array, sfreq=sampling_frequency, first_samp=first_sample, event_id=event_id)
else:
    events_exist = False
    print("No events to plot.")
  
# Show the plot of selected signals and sleep stages
plt.show()
