from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from pyspark.sql import functions as F
from pyspark.sql.types import BinaryType
from awsglue.job import Job
from io import BytesIO
from PIL import Image
import boto3
import sys

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'S3_SRC_BUCKET', 'S3_TARGET_BUCKET', 'S3_RAW_DATA_DIR', 'S3_TARGET_DATA_DIR'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

src_bucket = args['S3_SRC_BUCKET']
target_bucket = args['S3_TARGET_BUCKET']
raw_data = args['S3_RAW_DATA_DIR']
target_dir = args['S3_TARGET_DATA_DIR']

s3_client = boto3.client('s3')
paginator = s3_client.get_paginator('list_objects_v2') # list of filenames in s3 dir

def preprocess(img_data):
    img = Image.open(BytesIO(img_data))
    img = img.resize((544, 832), Image.NEAREST)
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer.getvalue()

def upload_to_s3(source_key, processed_image):
    if processed_image:
        s3_client = boto3.client('s3')
        filename = source_key[len(raw_data):].lstrip('/')
        target_key = f"{target_dir}/{filename}"
        print(f"target key: {target_key}")
        s3_client.put_object(
            Bucket=target_bucket,
            Key=target_key,
            Body=processed_image,
            ContentType='image/png'
        )
        return 'success'
    return 'failed'

image_files = []
for page in paginator.paginate(Bucket=src_bucket, Prefix=raw_data):
    if 'Contents' in page:
        for obj in page['Contents']:
            source_key = obj['Key']
            if source_key.lower().endswith(('.png', '.jpg', '.jpeg')):
                response = s3_client.get_object(Bucket=src_bucket, Key=source_key)
                image_bytes = response['Body'].read()
                image_files.append({
                    'source_key': source_key,
                    'image_data': image_bytes
                })

print(f'Number of images: {len(image_files)}')

image_df = spark.createDataFrame(image_files)

processed_df = image_df.withColumn(
    'processed_image',
    F.udf(lambda img_data: preprocess(img_data), BinaryType())('image_data')
)

results_df = processed_df.withColumn(
    'status', 
    F.udf(lambda source_key, processed_image: upload_to_s3(source_key, processed_image), returnType=F.StringType())(
        'source_key', 'processed_image'
    )
)

total = results_df.count()
ok = results_df.filter(F.col('status') == 'success').count()
failed = total - ok

print(f'Total: {total}, Success: {ok}, Failed: {failed}')

job.commit()
