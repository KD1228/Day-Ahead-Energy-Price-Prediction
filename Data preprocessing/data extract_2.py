import zipfile
import os

# Define base directory where ZIP files are located and extracted
base_folder = r"C:\Users\kanch\Energy price prediction\Energy price prediction"
zip_folders = {
    "solar": os.path.join(base_folder, "DWD_solar_zips"),
    "wind": os.path.join(base_folder, "DWD_wind_zips"),
    "temp": os.path.join(base_folder, "DWD_temp_zips"),
}

# Define extraction directory
extract_folder = os.path.join(base_folder, "Extracted_Data")
os.makedirs(extract_folder, exist_ok=True)

# Function to extract ZIP files and save each dataset to a separate folder
def extract_zip_files(zip_folder, dataset_name):
    dataset_extract_folder = os.path.join(extract_folder, dataset_name)
    os.makedirs(dataset_extract_folder, exist_ok=True)

    extracted_any = False  # Flag to check if any files were extracted

    if not os.path.exists(zip_folder):
        print(f"⚠️ Warning: Folder not found for {dataset_name}: {zip_folder}")
        return dataset_extract_folder

    zip_files = [f for f in os.listdir(zip_folder) if f.endswith(".zip")]

    if not zip_files:
        print(f"⚠️ No ZIP files found in {zip_folder} for {dataset_name}")
        return dataset_extract_folder

    for zip_file in zip_files:
        zip_path = os.path.join(zip_folder, zip_file)
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_extract_folder)
                print(f"✅ Extracted: {zip_file} ({dataset_name})")

                extracted_any = True
        except zipfile.BadZipFile:
            print(f"⚠️ Corrupt ZIP file: {zip_file}")

    if not extracted_any:
        print(f"⚠️ No files were extracted for {dataset_name}. Check ZIP contents.")

    return dataset_extract_folder

# Extract data for Solar, Wind, and Temperature and save to their respective folders
for dataset_name, zip_folder in zip_folders.items():
    print(f"\n🚀 Extracting {dataset_name.upper()} data from {zip_folder} ...")
    extract_zip_files(zip_folder, dataset_name)

print("\n🎉 Extraction completed! Data is saved into respective folders in Extracted_Data.")
