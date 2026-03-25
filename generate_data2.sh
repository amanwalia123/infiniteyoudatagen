#!/bin/sh
# This script is used to generate data for the InfiniteYou project.
# It sets up the environment and runs the data generation script with specified arguments.
# Ensure the script is run with the correct Python interpreter
# and that the necessary dependencies are installed.

# Set the Python interpreter to use
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv $VENV_DIR
fi

source $VENV_DIR/bin/activate
echo "Installing requirements..."
pip install -r requirements.txt

# Run the data generation script with the specified arguments
python data_generator.py \
    --output_dir /netapp/output/aman.walia/data/infiniteyou_face_dataset \
    --model_version sim_stage1 \
    --enable_anti_blur_lora \
    --cuda_device 2 \
    --num-samples 100 \
    --num-repeat 1