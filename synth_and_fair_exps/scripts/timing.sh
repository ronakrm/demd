data='celeba'
batch_size=64
learning_rate=0.001
outfile='results/timing_celeba_redux.csv'
model='resnet18' # approx 1.6GB with batchsize 32

run () {
	python run.py  \
			--n_classes 1 \
			--nSens 2 \
			--epochs 1 \
			--lambda_reg 0.1 \
			--dataset $data  \
			--model $model \
			--batch_size $batch_size \
			--outfile $outfile \
			--nbins $1 \
			--train_seed $2 \
			--regType $3 
}

for seed in 1 2 3 4 5 6 7 8 9
do
	# for n in 2 5 10 20 50 100
	# do
	# 	run $n $seed 'demd'
	# done
	# run 10 $seed 'none'
	# run 10 $seed 'dp'
	run 10 $seed 'eo'
	# run 10 $seed 'wasbary'
done