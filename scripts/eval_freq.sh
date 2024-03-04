#!/bin/bash


topk="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"


# Evaluate the MoE model on the test set
for k in $topk
    do
        echo "Evaluating MoE model with topk = $k to calculate frequency of expert selection"
        python moefication/freq_expert_select.py $k
    done
