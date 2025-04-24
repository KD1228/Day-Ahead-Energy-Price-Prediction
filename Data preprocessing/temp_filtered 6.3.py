import pandas as pd
import os


# Define the path to the filtered CSV file
filtered_file_path = r"C:\Users\kanch\Energy price prediction\Energy price prediction\Filtered_data\temp\temp_filtered.csv"

# Read the filtered data from the CSV file with the correct delimiter (semicolon ';')
filtered_df = pd.read_csv(filtered_file_path)
filtered_df_copy = filtered_df.copy()

# Display the first few rows of the filtered DataFrame
print("First few rows of the filtered data:",filtered_df_copy.head(30))
# Optionally, display basic info about the DataFrame (e.g., shape, columns, data types)
print("\nBasic info about the filtered data:",filtered_df_copy.info())
# Display the number of rows in the filtered DataFrame
print(f"\nNumber of rows in the filtered data: {filtered_df_copy.shape[0]}")
# Check for any missing values in the dataset
missing_values = filtered_df_copy.isnull().sum()
print("\nMissing values in each column:",missing_values)
# Display basic statistics of numerical columns
print("\nBasic statistics of the numerical columns:",filtered_df_copy.describe())



# 1. Convert 'MESS_DATUM' to datetime format
filtered_df_copy['MESS_DATUM'] = pd.to_datetime(filtered_df_copy['MESS_DATUM'], errors='coerce')
# Verify the conversion and check the datatype
print("\nMESS DATUM DATATYPE AFTER CONVERSION",filtered_df_copy['MESS_DATUM'].dtype)  # Should show datetime64[ns] if conversion is successful



# 2. Remove Outliers
clean_temp_df = filtered_df_copy[~filtered_df_copy.isin([-999]).any(axis=1)]



#3. drop columns
drop_columns=["STATIONS_ID", "eor","QN_9"]
clean_temp_df = clean_temp_df.drop(columns= drop_columns)

# Strip spaces from the column names
clean_temp_df.columns = clean_temp_df.columns.str.strip()


#4. Check for irregular Values
print("RF TU GREATRE THAN 90 =",(clean_temp_df["RF_TU"] > 90).sum())


# 5 Groupby DATE COLUMN

temp_df_cleaned_new = clean_temp_df.copy()
temp_df_cleaned_new['MESS_DATUM'] = pd.to_datetime(temp_df_cleaned_new['MESS_DATUM'])
temp_df_cleaned_new["MESS_DATUM"] = temp_df_cleaned_new["MESS_DATUM"].dt.floor("H")

final_temp_reset = temp_df_cleaned_new.groupby("MESS_DATUM").mean().reset_index()



print("Head Final Data: ",final_temp_reset.head(30))
print("Describe Final Data: ",final_temp_reset.describe())
print("Info Final Data: ",final_temp_reset.info())


# 7. Save the final DataFrame (final_wind_df) into a CSV file
final_temp_reset.to_csv('final_temp_data.csv')
folder_path = 'FINAL DATA FOR EDA'  # You can modify this path

# Create the folder if it doesn't exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Define the full file path where the CSV will be saved
file_path = os.path.join(folder_path, 'final_temp_data.csv')

# Save the DataFrame into the folder
final_temp_reset.to_csv(file_path)

print(f"File saved at: {file_path}")