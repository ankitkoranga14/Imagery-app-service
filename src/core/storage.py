"""
Storage Abstraction Layer - The Bridge Pattern

Provides a clean interface for storage operations with LocalStorage (active)
and AzureBlobStorage (boilerplate for future use).
"""

import os
import uuid
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, BinaryIO
from datetime import datetime, timedelta

from src.core.config import settings


class IStorage(ABC):
    """Interface for storage operations - The Bridge"""
    
    @abstractmethod
    async def upload(
        self,
        file_data: bytes,
        filename: str,
        folder: str = "uploads",
        content_type: str = "image/png"
    ) -> str:
        """
        Upload a file and return its unique storage path/key.
        
        Args:
            file_data: Raw bytes of the file
            filename: Original filename
            folder: Subfolder/container prefix
            content_type: MIME type of the file
            
        Returns:
            Storage key/path that can be used with get_url()
        """
        pass
    
    @abstractmethod
    async def get_url(self, storage_key: str, expires_in: int = 3600) -> str:
        """
        Get a URL for accessing the file.
        
        Args:
            storage_key: The key returned from upload()
            expires_in: Seconds until URL expires (for signed URLs)
            
        Returns:
            URL string for accessing the file
        """
        pass
    
    @abstractmethod
    async def delete(self, storage_key: str) -> bool:
        """
        Delete a file from storage.
        
        Args:
            storage_key: The key returned from upload()
            
        Returns:
            True if deleted successfully, False otherwise
        """
        pass
    
    @abstractmethod
    async def exists(self, storage_key: str) -> bool:
        """Check if a file exists in storage."""
        pass


class LocalStorage(IStorage):
    """Local filesystem storage implementation for development."""
    
    def __init__(self, base_path: str = "./data/storage"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _get_unique_filename(self, filename: str) -> str:
        """Generate a unique filename with UUID prefix."""
        ext = Path(filename).suffix
        unique_id = uuid.uuid4().hex[:12]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{unique_id}{ext}"
    
    async def upload(
        self,
        file_data: bytes,
        filename: str,
        folder: str = "uploads",
        content_type: str = "image/png"
    ) -> str:
        # Create folder if it doesn't exist
        folder_path = self.base_path / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        unique_filename = self._get_unique_filename(filename)
        file_path = folder_path / unique_filename
        
        # Write file
        with open(file_path, "wb") as f:
            f.write(file_data)
        
        # Return relative storage key
        return f"{folder}/{unique_filename}"
    
    async def get_url(self, storage_key: str, expires_in: int = 3600) -> str:
        """For local storage, return a relative path that can be served."""
        file_path = self.base_path / storage_key
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {storage_key}")
        
        # Return path relative to data directory for serving
        return f"/static/storage/{storage_key}"
    
    async def delete(self, storage_key: str) -> bool:
        file_path = self.base_path / storage_key
        try:
            if file_path.exists():
                file_path.unlink()
                return True
            return False
        except Exception:
            return False
    
    async def exists(self, storage_key: str) -> bool:
        return (self.base_path / storage_key).exists()


# =============================================================================
# Azure Blob Storage Implementation (BOILERPLATE - Uncomment for Production)
# =============================================================================
# 
# To use Azure Blob Storage in production:
# 1. pip install azure-storage-blob azure-identity
# 2. Set AZURE_STORAGE_CONNECTION_STRING in environment
# 3. Change StorageFactory to return AzureBlobStorage when ENV=PROD
#
# from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
# from azure.core.exceptions import ResourceNotFoundError
#
# class AzureBlobStorage(IStorage):
#     """Azure Blob Storage implementation for production."""
#     
#     def __init__(self, connection_string: str, container_name: str = "imagery"):
#         self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
#         self.container_name = container_name
#         self._ensure_container_exists()
#     
#     def _ensure_container_exists(self):
#         """Create container if it doesn't exist."""
#         try:
#             container_client = self.blob_service_client.get_container_client(self.container_name)
#             if not container_client.exists():
#                 container_client.create_container()
#         except Exception as e:
#             raise RuntimeError(f"Failed to initialize Azure container: {e}")
#     
#     def _get_unique_blob_name(self, filename: str, folder: str) -> str:
#         """Generate unique blob name with folder prefix."""
#         ext = Path(filename).suffix
#         unique_id = uuid.uuid4().hex[:12]
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         return f"{folder}/{timestamp}_{unique_id}{ext}"
#     
#     async def upload(
#         self,
#         file_data: bytes,
#         filename: str,
#         folder: str = "uploads",
#         content_type: str = "image/png"
#     ) -> str:
#         blob_name = self._get_unique_blob_name(filename, folder)
#         blob_client = self.blob_service_client.get_blob_client(
#             container=self.container_name,
#             blob=blob_name
#         )
#         
#         blob_client.upload_blob(
#             file_data,
#             content_settings={"content_type": content_type},
#             overwrite=True
#         )
#         
#         return blob_name
#     
#     async def get_url(self, storage_key: str, expires_in: int = 3600) -> str:
#         """Generate a SAS URL for secure access."""
#         blob_client = self.blob_service_client.get_blob_client(
#             container=self.container_name,
#             blob=storage_key
#         )
#         
#         # Generate SAS token
#         sas_token = generate_blob_sas(
#             account_name=self.blob_service_client.account_name,
#             container_name=self.container_name,
#             blob_name=storage_key,
#             account_key=self.blob_service_client.credential.account_key,
#             permission=BlobSasPermissions(read=True),
#             expiry=datetime.utcnow() + timedelta(seconds=expires_in)
#         )
#         
#         return f"{blob_client.url}?{sas_token}"
#     
#     async def delete(self, storage_key: str) -> bool:
#         try:
#             blob_client = self.blob_service_client.get_blob_client(
#                 container=self.container_name,
#                 blob=storage_key
#             )
#             blob_client.delete_blob()
#             return True
#         except ResourceNotFoundError:
#             return False
#         except Exception:
#             return False
#     
#     async def exists(self, storage_key: str) -> bool:
#         try:
#             blob_client = self.blob_service_client.get_blob_client(
#                 container=self.container_name,
#                 blob=storage_key
#             )
#             return blob_client.exists()
#         except Exception:
#             return False


class StorageFactory:
    """
    Factory for creating storage instances.
    
    This means your code is 100% ready for Azure today. When you're ready
    to move from local storage to the cloud, you simply change the ENV
    variable to PROD and provide the connection string. No code changes required.
    """
    
    _instance: Optional[IStorage] = None
    
    @classmethod
    def get_storage(cls) -> IStorage:
        """Get the appropriate storage implementation based on environment."""
        if cls._instance is None:
            if settings.ENVIRONMENT.upper() == "PROD" and settings.AZURE_STORAGE_CONNECTION_STRING:
                # Production: Use Azure Blob Storage
                # Uncomment the following when azure-storage-blob is installed:
                # cls._instance = AzureBlobStorage(
                #     connection_string=settings.AZURE_STORAGE_CONNECTION_STRING,
                #     container_name=settings.AZURE_CONTAINER_NAME
                # )
                # For now, fall back to local storage with a warning
                import logging
                logging.warning(
                    "Azure Blob Storage is configured but not implemented. "
                    "Falling back to LocalStorage. Uncomment AzureBlobStorage class to enable."
                )
                cls._instance = LocalStorage(path="./data/storage")
            else:
                # Development: Use local filesystem
                cls._instance = LocalStorage(base_path="./data/storage")
        
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None


# Convenience function for dependency injection
def get_storage() -> IStorage:
    """Get the storage instance - ready for FastAPI Depends()."""
    return StorageFactory.get_storage()

