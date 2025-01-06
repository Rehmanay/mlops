import os
import json
import boto3
import time

s3 = boto3.client('s3')
glue = boto3.client('glue')
cp = boto3.client('codepipeline')

def wait_for_glue_job(job_name, run_id):
    """Poll Glue job status until it is finished."""
    while True:
        response = glue.get_job_run(JobName=job_name, RunId=run_id, PredecessorsIncluded=False)
        status = response['JobRun']['JobRunState']
        if status in ['SUCCEEDED']:
            print(f"Glue job {job_name} completed successfully.")
            return True
        elif status in ['FAILED', 'STOPPED', 'TIMEOUT']:
            print(f"Glue job {job_name} failed with status: {status}.")
            raise Exception(f"Glue job {job_name} failed with status: {status}.")
        time.sleep(30)  

def lambda_handler(event, context):
    bucket_name = event.get('bucket_name', '<bucket-name>')
    file_key = event.get('json_location', 'etl/etl.json')

    src_bucket = event.get('--S3_SRC_BUCKET', '<bucket-name>')
    tar_bucket = event.get('--S3_TARGET_BUCKET', '<bucket-name>')
    raw_dir = event.get('--S3_RAW_DATA_DIR', 'raw_data') 
    tar_dir = event.get('--S3_TARGET_DATA_DIR', 'training/data')

    jobId = event['CodePipeline.job']['id']

    try:
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        file_content = response['Body'].read().decode('utf-8')
        etlJob = json.loads(file_content)

        # Start Glue job 1
        job_name_1 = 'img-preprocessing'
        etlJob['Name'] = job_name_1
        glue.create_job(**etlJob)
        response = glue.start_job_run(
            JobName=job_name_1,
            Arguments={
                "--JOB_NAME": job_name_1,
                "--S3_SRC_BUCKET": src_bucket,
                "--S3_TARGET_BUCKET": tar_bucket,
                "--S3_RAW_DATA_DIR": os.path.join(raw_dir, 'images'),
                "--S3_TARGET_DATA_DIR": os.path.join(tar_dir, 'images')
            }
        )
        run_id_1 = response['JobRunId']

        # Start Glue job 2
        job_name_2 = 'mask-preprocessing'
        etlJob['Name'] = job_name_2
        glue.create_job(**etlJob)
        response = glue.start_job_run(
            JobName=job_name_2,
            Arguments={
                "--JOB_NAME": job_name_2,
                "--S3_SRC_BUCKET": src_bucket,
                "--S3_TARGET_BUCKET": tar_bucket,
                "--S3_RAW_DATA_DIR": os.path.join(raw_dir, 'masks'),
                "--S3_TARGET_DATA_DIR": os.path.join(tar_dir, 'masks')
            }
        )
        run_id_2 = response['JobRunId']
        
        wait_for_glue_job(job_name_1, run_id_1)
        wait_for_glue_job(job_name_2, run_id_2)

        # Signal success to CodePipeline
        cp.put_job_success_result(jobId=jobId)

    except Exception as e:
        # Signal failure to CodePipeline
        cp.put_job_failure_result(
            jobId=jobId,
            failureDetails={
                'type': 'ConfigurationError',
                'message': str(e),
                'externalExecutionId': context.aws_request_id
            }
        )

    return 'Done'