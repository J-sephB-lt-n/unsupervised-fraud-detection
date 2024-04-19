#!/bin/bash
#   Example usage:
#       $ bash data/analysis/view_random_n_rows.sh 10 data/kaggle_alaxi_paysim1.csv 

nrows=$1
filepath=$2

head -n 1 $2
shuf -n $1 $2
