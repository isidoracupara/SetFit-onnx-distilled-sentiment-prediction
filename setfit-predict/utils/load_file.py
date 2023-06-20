from azure.storage.blob import BlobServiceClient
import os

def load_model(blob_name: str) -> None:
    """
    Load model from blob storage
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
    
    print(f"Loading model....")
    
    model = blob_client.download_blob().readall()

    print("Model loaded!")

    return model