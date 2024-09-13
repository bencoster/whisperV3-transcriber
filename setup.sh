#!/bin/bash

# Exit on any error
set -e

# Create a virtual environment if it doesn't exist
if [ ! -d "env" ]; then
  echo "Creating virtual environment..."
  python3 -m venv env
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required Python packages
echo "Installing required Python packages..."
pip install -r requirements.txt

# Set executable permissions for all necessary scripts
echo "Setting executable permissions for scripts..."
chmod +x scripts/setup.sh
chmod +x scripts/run.sh
chmod +x src/install_cuda_nvcc.py
chmod +x src/transcribe.py

# If there are additional scripts that need execution permissions, add them here
# chmod +x path/to/other_script.sh

# Deactivate the environment after setup is done
deactivate

echo "Setup complete. To activate the environment manually, run 'source env/bin/activate'."
