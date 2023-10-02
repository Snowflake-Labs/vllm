#!/bin/bash

# List of models to test
MODELS=(
    # "TheBloke/Llama-2-7B-AWQ"
    # "mosaicml/mpt-7b"
    # "tiiuae/falcon-7b"
    # "codellama/CodeLlama-7b-hf"
    # "EleutherAI/gpt-j-6b"
    # "bigscience/bloom-7b1"
)


for model in "${MODELS[@]}"
do
    echo "Testing model: $model"
    # Remove / in model names
    model_name=${model//\//_}
    python tests/models/accuracy/test_accuracy.py $model > "acc/$model_name.txt"
done
