import csv

def save_results_to_csv(results, file_path):
    """
    Save the RQA results to a CSV file.

    Parameters:
    results (list of dict): List of dictionaries containing RQA results for each segment.
    file_path (str): Path to the CSV file where the results will be saved.

    This function writes the list of dictionaries to a CSV file. Each dictionary in the list
    represents the results of RQA for a specific segment and embedding dimension.
    """
    if not results:
        raise ValueError("The results list is empty. No data to save.")
    
    keys = results[0].keys() # Get the keys from the first dictionary as the header for the CSV

    with open(file_path, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys) # Create a DictWriter object with the specified fieldnames (CSV headers)
        dict_writer.writeheader()
        dict_writer.writerows(results)

    print(f"Results saved to {file_path}")