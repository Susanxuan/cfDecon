#!/bin/bash

# Usage: ./run.sh <input_path> <output_dir> <train_or_test>

# Check if the number of parameters is correct
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <input_path> <output_dir> <train_or_test (train/test)>"
    exit 1
fi

# Parameters
INPUT_PATH=$1  # Data input path
OUTPUT_DIR=$2  # Output directory
MODE=$3        # Training mode (train or test)
USE_NORM="False" # Whether to use normalization (True/False)
SIMI=1         # Similarity type
NUM_SAMPLES=9  # Number of samples
NUM_CELL=9     # Number of cell types
ALPHA=10       # Alpha value
BETA=1         # Beta value
PROPORTIONS="./results/fractions/sample_array_10009_spar0.1False0.4.pkl"  # Predefined proportions path
R1=(128 72 48 12)  # Encoder parameters
R2=(12 64 128)     # Decoder parameters
FILE="0.1False0.4" # File name
N=5000             # Number of training samples
FIX="True"         # UXM mode

# Set train and test options based on the selected mode
if [ "$MODE" == "train" ]; then
    TRAIN=1
    TEST=0
elif [ "$MODE" == "test" ]; then
    TRAIN=0
    TEST=1
else
    echo "Invalid mode. Please choose either 'train' or 'test'."
    exit 1
fi

# Run the Python script
python3 main.py \
  --input_path $INPUT_PATH \
  --output_dir $OUTPUT_DIR \
  --train $TRAIN \
  --test $TEST \
  --use_norm $USE_NORM \
  --alpha $ALPHA \
  --beta $BETA \
  --simi $SIMI \
  --num_samples $NUM_SAMPLES \
  --num_cell $NUM_CELL \
  --proportions $PROPORTIONS \
  --r1 "${R1[@]}" \
  --r2 "${R2[@]}" \
  --file $FILE \
  --n $N \
  --fix $FIX
