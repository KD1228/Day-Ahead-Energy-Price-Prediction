import os
import pandas as pd

# Define base paths
base_folder = r"C:\Users\kanch\Energy price prediction\Energy price prediction"
cleaned_folder = os.path.join(base_folder, "Cleaned_Data")  # Folder with cleaned data
merged_folder = os.path.join(base_folder, "Merged_Data")  # Folder to save merged data

# Ensure merged data folder exists
os.makedirs(merged_folder, exist_ok=True)


# Function to concatenate all .txt files in the wind dataset
def concat_wind_files():
    dataset_name = "wind"
    dataset_clean_folder = os.path.join(cleaned_folder, dataset_name)  # Path to wind data folder
    dataset_merged_folder = os.path.join(merged_folder, dataset_name)  # Path to save merged wind data
    os.makedirs(dataset_merged_folder, exist_ok=True)  # Ensure output folder exists

    all_files = [f for f in os.listdir(dataset_clean_folder) if f.endswith(".txt")]
    dataframes = []  # List to store DataFrames

    if not all_files:
        print(f"⚠️ No .txt files found in {dataset_clean_folder}!")
        return

    for file in all_files:
        file_path = os.path.join(dataset_clean_folder, file)

        try:
            df = pd.read_csv(file_path, sep=";", encoding="utf-8", low_memory=False)
            dataframes.append(df)  # Append each DataFrame to the list
            print(f"✅ Loaded: {file}")

        except Exception as e:
            print(f"❌ Error reading {file}: {e}")

    # Concatenate all DataFrames
    if dataframes:
        merged_df = pd.concat(dataframes, ignore_index=True)  # Stack all data
        merged_file_path = os.path.join(dataset_merged_folder, "wind_merged.txt")
        merged_df.to_csv(merged_file_path, sep=";", index=False, encoding="utf-8")
        print(f"🎉 Merged wind dataset saved at: {merged_file_path}\n")
    else:
        print("⚠️ No valid data found to concatenate!")


# Run concatenation only for wind data
print("\n🚀 Processing wind data ...")
concat_wind_files()
print("\n🎉 Wind dataset concatenated successfully!")
