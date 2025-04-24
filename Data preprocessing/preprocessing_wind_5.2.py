import pandas as pd
import os

# Path to your merged data file (replace with the actual path to your file)
file_path = r"C:\Users\kanch\Energy price prediction\Energy price prediction\Merged_Data\wind\wind_merged.txt"

# Read the file (assuming it's delimited by semicolons)
df = pd.read_csv(file_path, sep=";", encoding="utf-8")

# Display the first few rows of the dataframe
print(df.head(30))
print(df.info())
print(df.describe())
print(df.columns)



# Convert MESS_DATUM to datetime, handling invalid dates as NaT
df['MESS_DATUM'] = pd.to_datetime(df['MESS_DATUM'], format='%Y%m%d%H', errors='coerce')
print(df.head(30))
# Check how many invalid 'MESS_DATUM' values there are after conversion
invalid_dates = df[df['MESS_DATUM'].isna()]
print(f"Number of rows with invalid MESS_DATUM: {invalid_dates.shape[0]}")

# Filter the data for the date range from 1st January 2019 to 31st December 2024
start_date = '2019-01-01'
end_date = '2024-12-31'

# Make sure to check the min and max date in the dataset to ensure that the range is correct
min_date = df['MESS_DATUM'].min()
max_date = df['MESS_DATUM'].max()
print(f"Date range in the dataset: {min_date} to {max_date}")

# Apply the filter only if the dates are within the range
filtered_df_wind = df[(df['MESS_DATUM'] >= start_date) & (df['MESS_DATUM'] <= end_date)]

# Check the size of the filtered dataset
print(f"Rows after filtering: {filtered_df_wind.shape[0]}")

# Display the first few rows of the filtered dataframe
print(filtered_df_wind.head(50))

# Define the output directory and create it if it doesn't exist
output_directory = r"C:\Users\kanch\Energy price prediction\Energy price prediction\Filtered_data\wind"
os.makedirs(output_directory, exist_ok=True)

# Define the file path to save the filtered data
filtered_file_path = os.path.join(output_directory, "wind_filtered.csv")

# Save the filtered data to the new file
filtered_df_wind.to_csv(filtered_file_path, index=False)

# Confirm that the file was saved
print(f"Filtered data saved to: {filtered_file_path}")
