import os
import shutil

# Define the base directory where extracted data is stored
base_folder = r"C:\Users\kanch\My_Learnings\Machine Learning Energy Project"
extracted_folder = os.path.join(base_folder, "Extracted_Data")
cleaned_folder = os.path.join(base_folder, "Cleaned_Data")  # Folder where clean data will be saved
os.makedirs(cleaned_folder, exist_ok=True)


# Function to delete metadata files and save cleaned files
def delete_metadata_and_save_cleaned(dataset_name):
    dataset_extract_folder = os.path.join(extracted_folder, dataset_name)
    cleaned_dataset_folder = os.path.join(cleaned_folder, dataset_name)

    # Create the cleaned folder for each dataset (solar, wind, temp)
    os.makedirs(cleaned_dataset_folder, exist_ok=True)

    if not os.path.exists(dataset_extract_folder):
        print(f"⚠️ Folder not found: {dataset_extract_folder}")
        return

    deleted_files = []  # To keep track of deleted files
    saved_files = []  # To keep track of saved (non-metadata) files

    for root, dirs, files in os.walk(dataset_extract_folder):
        for file in files:
            file_path = os.path.join(root, file)

            if "Metadaten" in file:  # Check if "Metadaten" is in the file name
                os.remove(file_path)  # Delete the metadata file
                deleted_files.append(file_path)
                print(f"✅ Deleted metadata file: {file_path}")
            else:
                # Copy the valid file to the cleaned folder
                shutil.copy(file_path, cleaned_dataset_folder)
                saved_files.append(file_path)
                print(f"✅ Saved cleaned file: {file_path}")

    if not deleted_files:
        print(f"⚠️ No metadata files found to delete in {dataset_name}.")
    else:
        print(f"\n🎉 {len(deleted_files)} metadata file(s) deleted for {dataset_name}.")

    if not saved_files:
        print(f"⚠️ No valid files found to save in {dataset_name}.")
    else:
        print(f"🎉 {len(saved_files)} cleaned file(s) saved for {dataset_name}.")


# Delete metadata files and save the cleaned data for solar, wind, and temp folders
for dataset_name in ["solar", "wind", "temp"]:
    print(f"\n🚀 Processing {dataset_name} data ...")
    delete_metadata_and_save_cleaned(dataset_name)

print("\n🎉 Metadata deletion and clean data saving completed!")
