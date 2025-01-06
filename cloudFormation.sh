aws cloudformation create-stack \
    --stack-name TrainingPipe \
    --capabilities CAPABILITY_NAMED_IAM \
    --template-body file://training-pipe.yml
