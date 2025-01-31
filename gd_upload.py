from google.cloud import storage
from google.oauth2 import service_account
import os

class GDriveUploader:
    def __init__(self, bucket_path):
        try:
            # Split bucket name and base path
            bucket_parts = bucket_path.split('/', 1)
            self.bucket_name = bucket_parts[0]
            self.base_path = bucket_parts[1] if len(bucket_parts) > 1 else ''

            credentials = service_account.Credentials.from_service_account_file(
                'service-account-key.json',
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            self.storage_client = storage.Client(credentials=credentials)
            self.bucket = self.storage_client.bucket(self.bucket_name)
        except Exception as e:
            print(f"Error initializing GDriveUploader: {str(e)}")
            raise

    def upload_file(self, file_path, destination_name=None):
        """Upload a file to Google Cloud Storage bucket."""
        try:
            if destination_name is None:
                destination_name = os.path.basename(file_path)
            
            # Combine base path with destination name
            full_path = f"{self.base_path}/{destination_name}" if self.base_path else destination_name
            blob = self.bucket.blob(full_path)
            blob.upload_from_filename(file_path)
            
            return {
                'bucket_path': f"gs://{self.bucket_name}/{full_path}",
                'public_url': blob.public_url
            }
            
        except Exception as e:
            print(f"Error uploading {file_path}: {str(e)}")
            return None

    def cleanup_local_files(self, directory):
        """Remove local files after successful upload."""
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(directory)