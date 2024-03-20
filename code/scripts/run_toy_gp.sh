#!/bin/bash

for prob in ackley10 levy10 rastrigin10 hartmann6;
do
    for rs in 1 2 3 4 5;
    do
        python toy_bo.py --problem $prob --randseed $rs --method gp --with_expert --expert_prob 0.1 --acqf_pref active_small_diff
    done
done
