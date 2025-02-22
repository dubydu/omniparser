import requests
from huggingface_hub import hf_hub_url
from pathlib import Path

# Set repository and file details
repo_id = "microsoft/OmniParser-v2.0"
filename = "icon_detect/model.pt"  # Replace this with the large file name
local_dir = "weights"
url = hf_hub_url(repo_id=repo_id, filename=filename)

# Specify the local path where the file will be saved
local_path = Path(local_dir) / filename

# Create directory if it doesn't exist
local_path.parent.mkdir(parents=True, exist_ok=True)

# Custom download with retries and timeout using requests
def download_file(url, local_path, retries=5, timeout=120):
    attempt = 0
    while attempt < retries:
        try:
            print(f"Attempting to download {filename}...")
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()  # Check for any HTTP errors
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Download complete: {local_path}")
            return local_path
        except requests.exceptions.RequestException as e:
            print(f"Error occurred: {e}")
            attempt += 1
            print(f"Retrying... ({attempt}/{retries})")
    raise Exception("Download failed after multiple retries.")

# Start the download
download_file(url, local_path)
