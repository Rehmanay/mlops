FROM nvidia/cuda:11.8.0-base-ubuntu20.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

    
WORKDIR /opt/ml/code
RUN mkdir -p /opt/ml/input/config
RUN mkdir -p /opt/ml/model
RUN mkdir -p /opt/ml/output
RUN mkdir -p /opt/ml/code

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY loader.py .
COPY seg_unet.py .
COPY train.py .

RUN chmod +x train.py
ENV SAGEMAKER_PROGRAM=train.py

ENTRYPOINT ["python3", "train.py"]