#!/bin/bash
echo 'Running Creating Data for Histogram convegence over DEMD...'

seed=1234
n=50
d=4
lr=0.00001
OUTPUT_FILE="results/1d_hist_results.csv"

run () {
	python test1d_opt.py --seed $seed \
					-n $n \
					-d $d \
					--iters $1 \
					--learning_rate $lr \
					--outfile $OUTPUT_FILE
}

run 0
run 10
run 50
run 100
run 200
run 500
run 1000