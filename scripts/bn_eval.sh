#!/usr/bin/env bash

set -e

echo $0
echo "Started"
date

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dagnn

PROJECT=$PWD
cd dvae/bayesian_optimization

export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH=$PYTHONPATH:$PROJECT:$PROJECT/dvae/bayesian_optimization/Theano-master/

echo "CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES}"
echo "PYTHONPATH ${PYTHONPATH}"
echo MODEL $2

# TODO SET THIS
DATAD=../bnresults/

MODEL=$2
NAME=$MODEL
CHECK=$3
RESULTS="${PROJECT}/bneval${CHECK}/"
mkdir -p $RESULTS

LAYERS=2
AGG=attn_h
POOL_ALL=0
POOL=max
DROPOUT=0
BIDIR=1
CLIP=0.25

if [[ "$MODEL" = "DAGNN"* ]]; then
    NAME="${MODEL}_l${LAYERS}_b${BIDIR}_a${AGG}_pa${POOL_ALL}_p${POOL}_c${CLIP}"
fi


python bo.py \
  --data-dir=$DATAD  \
  --data-name="asia_200k" \
  --save-appendix=$NAME --model=$MODEL --bidirectional \
  --checkpoint=$CHECK \
  --res-dir=$RESULTS \
  --dagnn_layers $LAYERS --dagnn_agg $AGG --dagnn_out_wx $OUT_WX \
  --dagnn_out_pool_all $POOL_ALL --dagnn_out_pool $POOL --dagnn_dropout $DROPOUT &> $RESULTS/"${NAME}.txt"

python summarize.py \
  --data-type=BN \
  --name=$NAME \
  --res-dir=$RESULTS &> $RESULTS/"${NAME}.txt"

echo "Completed"
date
