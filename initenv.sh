#!/bin/bash

# Stop the script immediately if any command fails
set -e 

echo "Starting setup..."
# 1. Check if the virtual environment exists. If not, create it.
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi
# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Upgrade pip to avoid "no matching distribution" errors
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# 4. Install your packages 
# (It's best to use a requirements.txt file, but we will list them directly for now)
echo "Installing dependencies..."
python3 -m pip install numpy
python3 -m pip install numpy torch
python3 -m pip install polars   
python3 -m pip install kagglehub
python3 -m pip install scikit-learn

# 5. Run your actual Python file
echo "Running ..."
python3 transform-ton.py