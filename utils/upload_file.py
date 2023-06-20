from azure.storage.blob import BlobServiceClient
import os
import json

def upload_file(local_file_path: str, blob_name: str) -> None:
    """
    Upload a blob (file) to an Azure Storage container.
    :param local_file_path: The path to the local file to upload.
    :param blob_name: The name to give the blob (file) in the container.
    """
    
    # Load environment variables
    connection_string = os.getenv("AZURE_CONNECTION_STRING")
    container_name = os.getenv("STORAGE_CONTAINER")

    # Raise an error if the environment variables are not found
    if not connection_string or not container_name:
        raise ValueError("Missing environment variables... Please check your .env file")

    # Initialize the connection to Azure
    blob_service_client =  BlobServiceClient.from_connection_string(connection_string)
    
    # Create blob (file) with same name as local file name (But you )
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    
    print(f"uploading file: {local_file_path}....")
    
    # Open local file
    with open(local_file_path, "rb") as file:
      # Write local file to blob overwriting any existing data
      blob_client.upload_blob(file, overwrite=True)

    print("file uploaded!")

    return