"Download the UniCeption format checkpoints from the AirLab Data Server"

import argparse
import os

from minio import Minio
from minio.error import S3Error
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Download UniCeption format checkpoints from AirLab Data Server")
    parser.add_argument(
        "--folders",
        nargs="+",
        default=["encoders", "info_sharing", "prediction_heads", "examples"],
        help="List of folders to download (default: all folders). Choices: encoders, info_sharing, prediction_heads, examples",
    )
    parser.add_argument("--destination", type=str, default="./", help="Destination folder for downloaded checkpoints")
    args = parser.parse_args()

    access_key = "bT79gQYtfhpxFIitlpns"
    secret_key = "g7mSvUJ5k2a9mKv9IbhwXmUQjQX52MLwulhW9ONO"
    client = Minio("airlab-share-02.andrew.cmu.edu:9000", access_key=access_key, secret_key=secret_key, secure=True)

    bucket_name = "uniception"

    def download_folder(folder_name, bucket_name, client, destination_folder):
        folder_name = f"checkpoints/{folder_name}/"
        objects = client.list_objects(bucket_name, prefix=folder_name, recursive=True)
        for obj in tqdm(objects, desc=f"Downloading {folder_name}"):
            destination_file = os.path.join(destination_folder, obj.object_name)
            if not os.path.exists(destination_file):
                os.makedirs(os.path.dirname(destination_file), exist_ok=True)
                try:
                    client.fget_object(bucket_name, obj.object_name, destination_file)
                    print(f"Downloaded {obj.object_name} to {destination_file}")
                except S3Error as e:
                    print(f"Error downloading {obj.object_name}: {e}")
            else:
                print(f"File {destination_file} already exists. Skipping...")

    for folder in args.folders:
        download_folder(folder, bucket_name, client, args.destination)


if __name__ == "__main__":
    main()
