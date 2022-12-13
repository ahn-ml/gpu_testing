#!/usr/bin/env bash

# Check if the folder exists
if [ -d /tmp/gpu_testing ]; then
  # Delete the folder and its contents
  rm -rf /tmp/gpu_testing
fi

# Set the working directory
cd /tmp/

# Clone the repo
git clone https://github.com/ahn-ml/gpu_testing.git

# Move into the repo directory
cd gpu_testing

# Create the virtual environment
python3 -m venv test_gpu_env

# Activate the virtual environment
source test_gpu_env/bin/activate

# Install the required packages
pip install -r requirements.txt

# Run the test script
python test_gpu.py

# Deactivate the virtual environment
deactivate

# Remove the repo and virtual environment
rm -rf /tmp/gpu_testing
