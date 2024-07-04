import os
from data_processing import process_file

def main():
    # List of ABF files to be analyzed
    file_list = [
        "C:\\Users\\grant\\OneDrive - Texas A&M University\\D2 1mm TBI control\\abf-files\\2023_09_06_0007.abf",
        # Add more file paths as needed
    ]
    
    # Create output folder in a subfolder called "outputs"
    script_directory = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_directory, "outputs")
    
    target_sampling_rate = 400  # Target downsampling rate in Hz
    epoch_duration_seconds = 12  # Duration of each epoch in seconds
    embedding_dims = [5, 9, 13, 17, 21, 25]

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each ABF file
    for file_path in file_list:
        process_file(file_path, output_folder, target_sampling_rate, epoch_duration_seconds, embedding_dims)

if __name__ == "__main__":
    main()
