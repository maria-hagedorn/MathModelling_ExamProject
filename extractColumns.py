import pandas as pd
import glob
import os

os.chdir('code')
print("Current working directory:", os.getcwd())

# Specify the directory containing the Excel files and the path for the CSV files
excel_files = glob.glob('data/*.xlsx')
output_directory_path = 'data/csv_files/'

print("Found Excel files:", excel_files)

# Create the output directory if it doesn't exist
#if not os.path.exists(output_directory_path):
#    os.makedirs(output_directory_path)

# Process each Excel file
for excel_file in excel_files:
    # Load the Excel file
    df = pd.read_excel(excel_file)
    
    # Exclude the last 6 columns (if there are at least 6 columns)
    df_trimmed = df.iloc[:, :-6] if df.shape[1] > 6 else df
    
    # Construct the output CSV file path
    base_name = os.path.basename(excel_file)  # Extracts the file name from the path
    csv_file_name = f"{os.path.splitext(base_name)[0]}.csv"  # Replaces the extension with .csv
    csv_file_path = os.path.join(output_directory_path, csv_file_name)  # Saves the CSV in the same directory
    
    # Save the transformed data to a CSV file
    df_trimmed.to_csv(csv_file_path, index=False)
    
    print(f"Processed and saved: {csv_file_path}")

# Notify when all files have been processed
print("All Excel files have been processed.")
