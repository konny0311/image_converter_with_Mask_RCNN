import cv2
from google.cloud import storage

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    img = cv2.imread(source_file_name)
    _, img_str = cv2.imencode('.jpg', img)
    img_bytes = img_str.tobytes()
    # blob.upload_from_filename(source_file_name)
    blob.upload_from_string(img_bytes)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )
def implicit():
    from google.cloud import storage

    # If you don't specify credentials when constructing the client, the
    # client library will look for credentials in the environment.
    storage_client = storage.Client()

    # Make an authenticated API request
    buckets = list(storage_client.list_buckets())
    print(buckets)

upload_blob('line-konny', 'test.jpg', 'dogdog.jpg')
# implicit()