"""
S3 / MinIO client helpers.

Centralizes connection setup so the rest of the codebase
doesn't repeat boto3 boilerplate.
"""

from functools import lru_cache
from pathlib import Path

import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError

from fundlens.config import settings


@lru_cache(maxsize=1)
def get_s3_client() -> BaseClient:
    """Return a singleton boto3 S3 client configured for MinIO."""
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key.get_secret_value(),
        region_name=settings.s3_region,
    )


def download_to_local(bucket: str, key: str, local_path: Path) -> Path:
    """
    Download an S3 object to a local file.
    Returns the local path.
    """
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3 = get_s3_client()
    s3.download_file(Bucket=bucket, Key=key, Filename=str(local_path))
    return local_path


def upload_file(local_path: Path, bucket: str, key: str, content_type: str = None) -> str:
    """
    Upload a local file to S3 / MinIO.
    Returns the S3 key.
    """
    s3 = get_s3_client()
    extra_args = {}
    if content_type:
        extra_args["ContentType"] = content_type
    s3.upload_file(
        Filename=str(local_path),
        Bucket=bucket,
        Key=key,
        ExtraArgs=extra_args,
    )
    return key


def object_exists(bucket: str, key: str) -> bool:
    """Check whether an S3 object exists."""
    try:
        get_s3_client().head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def ensure_bucket(bucket: str) -> None:
    """Create bucket if it doesn't already exist."""
    s3 = get_s3_client()
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError:
        s3.create_bucket(Bucket=bucket)