#!/bin/bash

declare -a problems=("ackley10constrained" "levy10constrained")
declare -a methods=("la" "gp")
declare -a expert_probs=(0.1 0.25)
declare -a randseeds=(1 2 3 4 5)

for problem in "${problems[@]}";
do
    for method in "${methods[@]}";
    do
        for prob in "${expert_probs[@]}";
        do
            for randseed in "${randseeds[@]}";
            do
                python toy_bo.py --problem $problem --randseed $randseed --method $method --with_expert --expert_prob $prob
            done
        done
    done
done
