#!/bin/bash

# beware, he be dragons
# only after training if fully completed
lr=3e-4
loss=CE

# stats on CV
python3 ./stats.py --cohort fit --model resnet34 --lr $lr > results/eval_final.txt

# retrain final model
python3  ./train.py --fold all --lr $lr --model resnet34  --cv 5  --loss $loss> results/train_final_$lr.log.txt

# generate data before computing stats
python3 ./validate.py --cohort ival --model resnet34 --cv 5 --lr $lr --loss $loss> results/val_final_ival.txt
python3 ./stats.py --cohort ival --model resnet34  --lr $lr --loss $loss> results/eval_ival.txt

python3 ./validate.py --cohort eli.2021A --model resnet34 --cv 5 --lr $lr --loss $loss > results/val_eli.2021A.txt
python3 ./stats.py --cohort eli.2021A --model resnet34 --lr $lr --loss $loss  > results/eval_eli.2021A.txt


#
