import os
import urllib.request
import zipfile
import yaml
from pathlib import Path

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def download_file(url, destination):
    print(f"Downloading {url} to {destination}...")
    def report_hook(block_num, block_size, total_size):
        read_so_far = block_num * block_size
        if total_size > 0:
            percent = read_so_far * 100 / total_size
            print(f"Downloaded {percent:.1f}% of {total_size / (1024 * 1024):.1f} MB", end="\r")
        else:
            print(f"Downloaded {read_so_far / (1024 * 1024):.1f} MB", end="\r")
    
    urllib.request.urlretrieve(url, destination, reporthook=report_hook)
    print("\nDownload complete.")

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")

def main():
    config = load_config()
    raw_dir = Path(config["paths"]["raw_data_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset URLs
    datasets = {
        "GTSRB_Train": {
            "url": "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip",
            "file": "GTSRB_Final_Training_Images.zip"
        },
        "GTSRB_Test": {
            "url": "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip",
            "file": "GTSRB_Final_Test_Images.zip"
        },
        "GTSRB_Test_GT": {
            "url": "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip",
            "file": "GTSRB_Final_Test_GT.zip"
        },
        "GTSDB": {
            "url": "https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/FullIJCNN2013.zip",
            "file": "FullIJCNN2013.zip"
        }
    }
    
    for name, info in datasets.items():
        zip_path = raw_dir / info["file"]
        extract_dest = raw_dir / name
        
        # Check if extracted directory already exists
        if extract_dest.exists():
            print(f"Directory {extract_dest} already exists, skipping download and extraction.")
            continue
            
        # Download zip
        if not zip_path.exists():
            try:
                download_file(info["url"], zip_path)
            except Exception as e:
                print(f"Failed to download {name}: {e}")
                continue
        else:
            print(f"{info['file']} already exists, skipping download.")
            
        # Extract zip
        try:
            extract_zip(zip_path, extract_dest)
        except Exception as e:
            print(f"Failed to extract {name}: {e}")

if __name__ == "__main__":
    main()
