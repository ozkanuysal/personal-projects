import unittest
from unittest.mock import patch, MagicMock
from minio.error import S3Error
from src.app.services.minio import MinioService
from minio import Minio as MockMinio
from src.app.crud.settings import settings as mock_settings

file_path = '~/notebooks/config.json'
file_name = "config.json"

class TestMinioService(unittest.TestCase):

    def setUp(self):
        # Set up the mock settings
        mock_settings.MINIO_HOST = 'localhost:9000'
        mock_settings.MINIO_ACCESS_KEY = 'MINIO_USER'
        mock_settings.MINIO_SECRET_KEY = 'MINIO_PASSWORD'
        mock_settings.MINIO_BUCKET_NAME = 'test-bucket'

        # Create a mock Minio client
        self.mock_minio_client = MagicMock()
        MockMinio.return_value = self.mock_minio_client

        # Initialize the MinioService
        self.service = MinioService()
        self.service.client = self.mock_minio_client

    def test_create_bucket_if_not_exists(self):
        # Test that the bucket creation is called when the bucket does not exist
        self.mock_minio_client.bucket_exists.return_value = False
        self.service.create_bucket_if_not_exists()
        self.mock_minio_client.make_bucket.assert_called_once_with('test-bucket')

    def test_upload_file(self):
        # Test that file upload is called correctly
        self.service.upload_file(file_path, file_name)
        self.mock_minio_client.fput_object.assert_called_once_with(
            'test-bucket', 'config.json', file_path
        )

    def test_download_file(self):
        # Test that file download is called correctly
        self.service.download_file('config.json', 'config.json')
        self.mock_minio_client.fget_object.assert_called_once_with(
            'test-bucket', file_name, file_name
        )

    def test_list_files(self):
        # Test listing files
        mock_objects = [MagicMock(object_name='file1.txt'), MagicMock(object_name='file2.txt')]
        self.mock_minio_client.list_objects.return_value = mock_objects
        files = self.service.list_files()
        self.assertEqual(files, ['file1.txt', 'file2.txt'])




if __name__ == '__main__':
    unittest.main()