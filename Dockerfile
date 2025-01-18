# Use the official Python image from Docker Hub
FROM python:3.11-slim

# Set the working directory to your project directory
WORKDIR /Users/chenxi/Desktop/02476/mlops_group_33

# Install necessary system dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc curl && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Install Google Cloud SDK
RUN curl https://sdk.cloud.google.com | bash > /dev/null && \
    /root/google-cloud-sdk/bin/gcloud components install beta

# Set environment variables for Google Cloud
ENV GOOGLE_APPLICATION_CREDENTIALS="dtumlops-448013-c1570ccce2fd.json"

# Copy the requirements file to the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire source code to the container
COPY . .

# Set the command to run your training script
ENTRYPOINT ["python", "-u", "src/final_project/train.py"]