#!/bin/bash

for prob in levy10 rastrigin10 hartmann6 ackley2;
do
    for rs in 1 2 3 4 5;
    do
        python toy_bo.py --problem $prob --randseed $rs --method la --with_expert --expert_prob 0.25
    done
done