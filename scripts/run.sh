#!/bin/bash

set -e

SEEDS=(1000 2000 3000 4000)

trial() {
  fname=$(basename "$1")
  id="${date,,}_${fname%.*}/seed=$2"

  echo -e "\nTraining for '$id'..."
  python openwindow/train.py -f "$1" --seed $2 --training_id "$id"

  echo -e "\nPredicting for '$id'..."
  python openwindow/predict.py -f "$1" --training_id "$id" --clean=True
}

evaluate() {
  echo "Evaluating '$1'..."
  fname=$(basename "$1")
  ids=${SEEDS[@]/#/${date,,}_${fname%.*}/seed=}
  python openwindow/evaluate.py -f "$1" --training_id ${ids// /,}
  echo
}

experiment() {
  echo "Running experiment for $(basename $1)..."
  for seed in ${SEEDS[@]}; do
    trial "$1" $seed
  done
  echo
}

date=$(date -u +%b%d)

# evaluate 'default.ini'
# evaluate 'scripts/closedset-50.ini'
# evaluate 'scripts/openset-o1-o2-f.ini'
experiment 'scripts/openset-o1-f-o2.ini'
evaluate 'scripts/openset-o1-f-o2.ini'
# evaluate 'scripts/openset-f-o1-o2.ini'
