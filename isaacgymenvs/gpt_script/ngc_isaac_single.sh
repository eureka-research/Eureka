#!/bin/bash

# the first command line argument is temperature, the second is sample, the third is iteration
TEMPERATURE=1.0
SAMPLE=16
ITERATION=5
MODEL=gpt-4-0314
JOB_NAME='reward-gpt4-debug-isaac-markov-gtfeedback'

python gpt_bidex.py --multirun env=$1 temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name=$JOB_NAME
