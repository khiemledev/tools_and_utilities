import argparse
import sys
from getpass import getpass
from pathlib import Path

import boto3
import boto3.session


def get_s3_client(access_key: str, secret_key: str, region: str | None = None):
    return boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        aws_session_token=None,
        config=boto3.session.Config(
            region_name=region if region else "ap-southeast-1",
            signature_version="s3v4",
        ),
        verify=True,
    )

def upload_to_s3(s3_client, file_path: str, object_key: str, bucket: str) -> bool:
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"Error: File {file_path} does not exist")
            return False

        print(f"Uploading {file_path} to S3...")
        s3_client.upload_file(str(file_path), bucket, object_key)
        print(f"Successfully uploaded file to s3://{bucket}/{object_key}")
        return True

    except Exception as err:
        print(f"Error uploading file to S3: {str(err)}")
        return False

def delete_from_s3(s3_client, object_key: str, bucket: str) -> bool:
    try:
        print(f"Deleting s3://{bucket}/{object_key}...")
        s3_client.delete_object(Bucket=bucket, Key=object_key)
        print(f"Successfully deleted s3://{bucket}/{object_key}")
        return True

    except Exception as err:
        print(f"Error deleting object from S3: {str(err)}")
        return False

def get_presigned_url(s3_client, object_key: str, bucket: str, expiration: int = 3600) -> bool:
    try:
        print(f"Generating presigned URL for s3://{bucket}/{object_key}...")
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': object_key},
            ExpiresIn=expiration
        )
        print(f"Presigned URL (valid for {expiration} seconds):")
        print(url)
        return True

    except Exception as err:
        print(f"Error generating presigned URL: {str(err)}")
        return False

def main():
    # Create main parser
    parser = argparse.ArgumentParser(
        description="S3 Utility Tool",
        epilog="""
Examples:
  # Upload a file
  python s3_util.py upload --file ./myfile.txt --obj_key folder/myfile.txt --bucket my-bucket

  # Generate presigned URL
  python s3_util.py presigned --obj_key folder/myfile.txt --bucket my-bucket --expiration 3600

  # Delete an object
  python s3_util.py delete --obj_key folder/myfile.txt --bucket my-bucket
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='action', help='Action to perform')

    # Add region argument to parent parser so it's available to all subcommands
    parser.add_argument("--region", help="AWS region (default: ap-southeast-1)")

    # Upload parser
    upload_parser = subparsers.add_parser('upload', help='Upload a file to S3')
    upload_parser.add_argument("--file", required=True, help="Path to the file to upload")
    upload_parser.add_argument("--obj_key", required=True, help="S3 object key (destination path)")
    upload_parser.add_argument("--bucket", required=True, help="S3 bucket name")

    # Delete parser
    delete_parser = subparsers.add_parser('delete', help='Delete an object from S3')
    delete_parser.add_argument("--obj_key", required=True, help="S3 object key to delete")
    delete_parser.add_argument("--bucket", required=True, help="S3 bucket name")

    # Presigned URL parser
    presigned_parser = subparsers.add_parser('presigned', help='Generate a presigned URL')
    presigned_parser.add_argument("--obj_key", required=True, help="S3 object key")
    presigned_parser.add_argument("--bucket", required=True, help="S3 bucket name")
    presigned_parser.add_argument("--expiration", type=int, default=3600, help="URL expiration in seconds (default: 3600)")

    args = parser.parse_args()

    if not args.action:
        parser.print_help()
        sys.exit(1)

    # Prompt for credentials
    access_key = input("ACCESS KEY: ").strip()
    secret_key = getpass("SECRET KEY: ").strip()

    # Create S3 client
    s3_client = get_s3_client(access_key, secret_key, args.region)

    # Execute requested action
    if args.action == 'upload':
        upload_to_s3(s3_client, args.file, args.obj_key, args.bucket)
    elif args.action == 'delete':
        delete_from_s3(s3_client, args.obj_key, args.bucket)
    elif args.action == 'presigned':
        get_presigned_url(s3_client, args.obj_key, args.bucket, args.expiration)

if __name__ == "__main__":
    main()
