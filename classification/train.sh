#!/bin/bash

#train first , fold by fold
rm -rf lightning_logs/
rm -rf results
mkdir results

for lr in 9e-4 6e-4 3e-4 1e-4 9e-5
do
    for f in 0 1 2 3 4
    do
        echo python3  ./train.py --fold $f --lr $lr --model resnet34  --cv 5 --loss CE> results/train_fold_${f}_lr_${lr}_CE.log.txt
        echo python3  ./train.py --fold $f --lr $lr --model resnet34  --cv 5 --loss focal> results/train_fold_${f}_lr_${lr}_focal.log.txt
    done
done

# validate internally
for lr in 9e-4 6e-4 3e-4 1e-4 9e-5
do
    python3 ./validate.py --cohort fit --model resnet34 --cv 5 --lr $lr --loss CE> results/log_val_lr_${lr}_CE.txt
    python3 ./validate.py --cohort fit --model resnet34 --cv 5 --lr $lr --loss focal > results/log_val_lr_${lr}_focal.txt
done

# look at logs now to see whos best-- stats_CV.txt
# then you are entitled to call ./eval.sh

#
