#!/bin/bash
#$ -M ltong2@nd.edu
#$ -m abe
#$ -q long

TRAIN_RAW='data/0518/full_short_0_-3/train.tsv'
TRAIN_INDEXED="${TRAIN_RAW%.*}.indexed.tsv"

less $TRAIN_RAW| awk -F '\t' '{print "0.0 "$1"\t1.0 "$2}'> $TRAIN_INDEXED

/afs/crc.nd.edu/user/l/ltong2/.conda/envs/LSP/bin/python prepro.py --corpus $TRAIN_INDEXED --max_seq_len 128