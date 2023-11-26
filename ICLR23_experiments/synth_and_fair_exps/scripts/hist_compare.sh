#!/bin/bash
echo 'Running Creating Data for Histogram convegence over DEMD...'

seed=1234
n=50
lr=0.00001
OUTPUT_FILE="results/hist_compare_results.csv"

run () {
	python hists_compare.py --seed $seed \
					-n $n \
					-d $1 \
					--baryType $2 \
					--iters 1000 \
					--learning_rate $lr \
					--outfile $OUTPUT_FILE
}

runinit () {
	python test1d_opt.py --seed $seed \
					-n $n \
					-d $1 \
					--iters 0 \
					--learning_rate $lr \
					--outfile "results/hist_init_results.csv"
}

for d in 2 5 10 20
do
	runinit $d
	# for baryType in 'lp' 'demd'
	# do
	# 	run $d $baryType
	# done
done