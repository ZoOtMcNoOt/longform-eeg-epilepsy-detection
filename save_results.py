import csv
import os

def save_results_to_csv(results, file_path):
    """
    Save the RQA results to a CSV file.

    Parameters:
    results (list of dict): List of dictionaries containing RQA results for each segment.
    file_path (str): Path to the CSV file where the results will be saved.

    This function appends the list of dictionaries to a CSV file if it exists. If the file
    does not exist, it creates a new file and writes the headers.
    """
    if not results:
        raise ValueError("The results list is empty. No data to save.")
    
    file_exists = os.path.isfile(file_path)
    keys = results[0].keys()  # Get the keys from the first dictionary as the header for the CSV

    with open(file_path, 'a', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)  # Create a DictWriter object with the specified fieldnames (CSV headers)
        if not file_exists:
            dict_writer.writeheader()  # Write the header only if the file does not exist
        dict_writer.writerows(results)

    print(f"Results saved to {file_path}")
