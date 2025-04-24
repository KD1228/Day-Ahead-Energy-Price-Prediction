import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Function to download ZIP files from a given URL
def download_data(base_url, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    response = requests.get(base_url)

    if response.status_code != 200:
        print(f"⚠️ Failed to access {base_url} (Status Code: {response.status_code})")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    zip_files = [link.get("href") for link in soup.find_all("a") if link.get("href", "").endswith(".zip")]

    if not zip_files:
        print(f"⚠️ No ZIP files found on {base_url}!")
        return []

    downloaded_files = []
    for file_name in zip_files:
        file_url = urljoin(base_url, file_name)
        file_path = os.path.join(save_folder, file_name)

        if not os.path.exists(file_path):  # Avoid re-downloading
            print(f"⬇️ Downloading {file_name}...")
            file_data = requests.get(file_url, stream=True)
            if file_data.status_code == 200:
                with open(file_path, "wb") as file:
                    for chunk in file_data.iter_content(chunk_size=1024):
                        file.write(chunk)
                print(f"✅ Saved: {file_path}")
            else:
                print(f"⚠️ Failed to download {file_name}")
        downloaded_files.append(file_path)

    return downloaded_files

# Define the data sources
data_sources = {
    "solar": {
        "urls": ["https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/solar/"],
        "zip_folder": "DWD_solar_zips"
    },
    "wind": {
        "urls": [
            "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/wind/historical/",
            "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/wind/recent/"
        ],
        "zip_folder": "DWD_wind_zips"
    },
    "air_temperature": {
        "urls": [
            "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/air_temperature/historical/",
            "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/air_temperature/recent/"
        ],
        "zip_folder": "DWD_temp_zips"
    }
}

# Download data for all categories
for category, info in data_sources.items():
    print(f"\n🚀 Downloading {category.upper()} data...")
    for url in info["urls"]:
        download_data(url, info["zip_folder"])

print("\n✅ All downloads completed!")
