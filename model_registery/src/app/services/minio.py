from minio import Minio
from minio.error import S3Error
from ..crud.settings import settings


class MinioService:
    """
    A service class for interacting with MinIO, a self-hosted object storage server.

    Attributes:
    - client: A Minio client instance for making requests to MinIO.
    - bucket_name: The name of the bucket to interact with.

    Methods:
    - __init__(self, bucket_name=None): Initializes the MinioService with a Minio client and a bucket name.
    - create_bucket_if_not_exists(self): Creates the bucket if it does not exist.
    - upload_file(self, file_obj, object_name): Uploads a file to MinIO.
    - download_file(self, object_name, file_path): Downloads a file from MinIO.
    - list_files(self, prefix=''): Lists all files in the bucket or with the given prefix.
    """

    def __init__(self, bucket_name=None):
        self.client = Minio(
            settings.MINIO_HOST,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=False,
        )
        self.bucket_name = bucket_name if bucket_name else settings.MINIO_BUCKET_NAME
        self.create_bucket_if_not_exists()

    def create_bucket_if_not_exists(self):
        """
        Creates the bucket if it does not exist.
        """
        if not self.client.bucket_exists(self.bucket_name):
            self.client.make_bucket(self.bucket_name)

    def upload_file(self, file_obj, object_name):
        """
        Uploads a file to MinIO.

        Parameters:
        - file_obj: The file to upload. It can be either a file path (str) or a file-like object.
        - object_name: The name to give the uploaded file in MinIO.

        Raises:
        - Exception: If there is an error uploading the file to MinIO.
        """
        try:
            # If the file_obj is a file path, use fput_object
            if isinstance(file_obj, str):
                self.client.fput_object(
                    self.bucket_name, object_name, file_obj
                )
            # If the file_obj is a file-like object, use put_object
            elif hasattr(file_obj, 'read'):
                file_obj.seek(0)  # Ensure we're at the start of the file-like object
                self.client.put_object(
                    self.bucket_name, object_name, file_obj, file_obj.getbuffer().nbytes
                )
            else:
                raise ValueError("file_obj must be a file path or file-like object")
        except S3Error as e:
            raise Exception(f"Failed to upload file to MinIO: {str(e)}")

    def download_file(self, object_name, file_path):
        """
        Downloads a file from MinIO.

        Parameters:
        - object_name: The name of the file in MinIO.
        - file_path: The local path where the downloaded file will be saved.

        Raises:
        - Exception: If there is an error downloading the file from MinIO.
        """
        try:
            self.client.fget_object(
                self.bucket_name, object_name, file_path
            )
        except S3Error as e:
            raise Exception(f"Failed to download {object_name} from MinIO: {str(e)}")

    def list_files(self, prefix=''):
        """
        List all files in the bucket or with the given prefix.

        Parameters:
        - prefix: The prefix to filter the files by.

        Returns:
        - A list of object names that match the given prefix.

        Raises:
        - Exception: If there is an error listing the files in MinIO.
        """
        try:
            objects = self.client.list_objects(self.bucket_name, prefix=prefix, recursive=True)
            return [obj.object_name for obj in objects]
        except S3Error as e:
            raise Exception(f"Failed to list files in MinIO: {str(e)}")


minio_service = MinioService()