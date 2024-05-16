#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <text_file>"
    exit 1
fi

# Assigning argument to a variable
text_file="$1"

# Check if the text file exists
if [ ! -f "$text_file" ]; then
    echo "Error: Text file '$text_file' not found."
    exit 1
fi

# Read each line of the text file and execute the command with the line as argument
while IFS= read -r line; do
    # Execute the command with the current line as an argument
    python benchmarks/object_erase.py --concepts_to_remove "$line" --gpu 7 --fine_tuned_unet union-timesteps --seed 0 --dataset_type modularity/datasets/imagenette.csv_keep
done < "$text_file"
