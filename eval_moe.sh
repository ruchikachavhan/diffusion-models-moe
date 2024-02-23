#!/bin/bash


# This script is used to evaluate the MoE model on the test set.
# Make a list of topk values
topk="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"


# Evaluate the MoE model on the test set
for k in $topk
    do
        echo "Evaluating MoE model with topk = $k"
        if [ $1 == "None" ]
        then
            python moefication/eval_moefied_sd.py --topk-experts $k --res-path results_1/stable-diffusion
        else
            python moefication/eval_moefied_sd.py --topk-experts $k --fine-tuned-unet $1 --res-path results_1/stable-diffusion
        fi
    done
