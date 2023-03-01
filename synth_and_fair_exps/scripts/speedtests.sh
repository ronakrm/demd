#!/bin/bash
echo 'Running Script to sweep parameters for Speed Test...'

OUTPUT_FILE="results/speed_test_results.csv"

startID=1
endID=10

run () {
	# echo $1 $2 $3 $4
	python speed_test.py --random_seed $1 \
					--n $2 \
					--d $3 \
					--gradType $4 \
					--outfile $OUTPUT_FILE
}

replicate() { 
	for runID in $(seq $startID 1 $endID)
		do
			run $runID $1 $2 $3 &
		done
}


for n in 2 5 10 20 50 100
do
	for d in 2 5 10 20 50 100
	do
		for gradType in 'scipy' 'npdual' 'autograd' 'torchdual'
		do
			replicate $n $d $gradType
		done
		wait
	done
done


