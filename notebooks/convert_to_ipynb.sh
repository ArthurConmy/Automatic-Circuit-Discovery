#!/bin/bash

# TODO add this all to the Makefile

set -e

# Check for --skip-run flag
skip_run=false
for arg in "$@"
do
    if [ "$arg" == "--skip-run" ]; then
        skip_run=true
    fi
done

required_commands=("jupytext" "papermill" "python")

# Loop over each required command
for command in "${required_commands[@]}"; do
    # Check if the command is installed
    if ! command -v $command &> /dev/null
    then
        echo "$command could not be found"
        echo "Please install it using 'pip install $command'"
        exit
    fi
done

# Define the file paths
declare -A file_paths
file_paths=(
    ["notebooks/editing_edges.py"]="notebooks/_converted/editing_edges.ipynb notebooks/colabs/ACDC_Editing_Edges_Demo.ipynb"
    ["acdc/main.py"]="notebooks/_converted/main_demo.ipynb notebooks/colabs/ACDC_Main_Demo.ipynb"
    ["notebooks/implementation_demo.py"]="notebooks/_converted/implementation_demo.ipynb notebooks/colabs/ACDC_Implementation_Demo.ipynb"
)

# Loop over each file path
for in_path in "${!file_paths[@]}"; do
    # Split the output paths
    IFS=' ' read -r -a out_paths <<< "${file_paths[$in_path]}"

    middle_path=${out_paths[0]}
    final_out_path=${out_paths[1]}

    # Run jupytext and papermill
    jupytext --to notebook "$in_path" -o "$middle_path"
    
    if ! $skip_run; then
        papermill "$middle_path" "$final_out_path" --kernel=python

        # TODO fix this; it seems some errored files are slipping through
        python -c "
import nbformat
nb = nbformat.read('$final_out_path', as_version=4)
errors = [cell for cell in nb['cells'] if 'outputs' in cell and any(output.get('output_type') == 'error' for output in cell['outputs'])]
if errors:
    raise Exception(f'Error: The following cells failed in notebook $final_out_path:\n{errors}')
"
    else
        cp "$middle_path" "$final_out_path"
    fi
done