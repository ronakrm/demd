#!/bin/bash
echo 'Running Script to compare NN backprop of Wass Barycenters...'

OUTPUT_FILE="results/wassimplresults.csv"

startID=1
endID=2
lr=0.01
n=100 # samples/batchsize
n_epochs=1000

run () {
	echo $1 $2 $3 $4
	python torchLayerTester.py --seed $1 \
					--nbins $2 \
					-d $3 \
					-n $n \
					--learning_rate $lr \
					--distType $4 \
					--n_epochs $n_epochs \
					--outfile $OUTPUT_FILE
}

replicate() { 
	for runID in $(seq $startID 1 $endID)
		do
			run $runID $1 $2 $3
		done
}


for nbins in 2 5 10 #20 50 100
do
	for d in 2 5 10 #20 50 100
	do
		for distType in 'demd' 'pairwass'
		do
			replicate $nbins $d $distType
		done
		wait
	done
done


