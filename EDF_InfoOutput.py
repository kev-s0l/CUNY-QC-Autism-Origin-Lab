import pyedflib

# Load the EDF file
edf_file = "C:/Users/kevin/Desktop/RESEARCH LAB/DummyData.edf"
edf = pyedflib.EdfReader(edf_file)

# Define number of sample points to use --> Adjustable
data_chunk_size = 10 # Here I wanted to use 10 

# Open a new file 
with open("edf_output_preview.txt", "w") as file:
    # Write basic EDF file information
    file.write("========== EDF File Information ==========\n")
    file.write(f"File Duration: {edf.file_duration} seconds\n")
    file.write(f"Start Date: {edf.getStartdatetime()}\n")
    file.write(f"Number of Signals: {edf.signals_in_file}\n\n")

    file.write("========== Signal Data ==========\n\n")

    # Loop through signals
    for i in range(edf.signals_in_file):
        channel_name = edf.getLabel(i)
        signal_data = edf.readSignal(i)

        # Write signal information with a small preview of data points
        file.write(f"Reading Signal {i}: {channel_name}\n")

        # Get the first 'data_chunk_size' data points
        data_chunk = signal_data[:data_chunk_size]
        file.write("Data chunk: [" + ", ".join(map(str, data_chunk)) + "]\n")

        # Write the total length of the signal
        file.write(f"Signal Length: {len(signal_data)}\n\n")

# Close file
edf.close()
