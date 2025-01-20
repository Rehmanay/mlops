import boto3
import json
import time

s3 = boto3.client('s3')
sm = boto3.client('sagemaker')
cp = boto3.client('codepipeline')

def lambda_handler(event, context):
    params = json.loads(event["CodePipeline.job"]["data"]["actionConfiguration"]["configuration"]["UserParameters"])
    bucket_name = params['bucket_name'] 
    file_key = params['json_location']
    tracking_uri = params['tracking_uri']
    job_name = f'lambda-sm-{str(time.time())[-5:]}'
    jobId = event['CodePipeline.job']['id']

    try:
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        file_content = response['Body'].read().decode('utf-8')
        tj = json.loads(file_content)
        tj['TrainingJobName'] = job_name
        tj['Environment']['TRACKING_URI'] = tracking_uri
        print(tj)

        sm.create_training_job(**tj)

        while True:
            training_job_status = sm.describe_training_job(TrainingJobName=job_name)
            status = training_job_status['TrainingJobStatus']

            if status == 'Completed':
                print(f"Training job {job_name} completed successfully.")
                cp.put_job_success_result(jobId=jobId)
                break
            elif status == 'Failed' or status == 'Stopped':
                error_message = training_job_status.get('FailureReason', 'No failure reason available')
                print(f"Training job {job_name} failed: {error_message}")
                cp.put_job_failure_result(
                    jobId=jobId,
                    failureDetails={
                        'type': 'JobFailed',
                        'message': error_message,
                        'externalExecutionId': context.aws_request_id
                    }
                )
                break
            else:
                print(f"Training job {job_name} is still in progress. Status: {status}")
            time.sleep(30)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        cp.put_job_failure_result(
            jobId=jobId,
            failureDetails={
                'type': 'ConfigurationError',
                'message': str(e),
                'externalExecutionId': context.aws_request_id
            }
        )
