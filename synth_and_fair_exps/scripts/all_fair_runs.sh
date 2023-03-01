epochs=20
learning_rate=0.01
model='ACSDeepNet'
nbins=10
outfile='results/res_all_fair_dsets_final.csv'

run () {
	python run.py  \
			--train_seed $3 \
			--dataset $data  \
			--model $model \
			--batch_size $batch_size \
			--input_size $input_size \
			--regType $2 \
			--n_classes $n_classes \
			--nbins $nbins \
			--nSens $nSens \
			--epochs $epochs \
			--lambda_reg $1 \
			--learning_rate $learning_rate \
			--outfile $outfile \
			--droplast True
}

full_loop () {


	seed=$1
	lamb=$2

	### German
	data='German'
	input_size=58
	batch_size=64
	n_classes=1
	nSens=1
	# outfile='results/german_class_res.csv'
	for reg in 'none' 'dp' 'eo' 'demd' 'wasbary'
	do
		run $lamb $reg $seed
	done

	### Adult
	data='adult'
	input_size=100
	batch_size=128
	n_classes=1
	nSens=1
	# outfile='results/adult_class_res.csv'
	for reg in 'none' 'dp' 'eo' 'demd' 'wasbary'
	do
		run $lamb $reg $seed
	done


	### Crime
	data='crime'
	input_size=245
	batch_size=64
	n_classes=1
	nSens=1
	# outfile='results/crime_class_res.csv'
	for reg in 'none' 'dp' 'eo' 'demd' 'wasbary'
	do
		run $lamb $reg $seed
	done

	### ACS Employment
	data='acs-employ'
	input_size=16
	batch_size=256
	n_classes=1
	nSens=9
	# outfile='results/acsemploy_class_res.csv'
	for reg in 'none' 'dp' 'eo' 'demd' 'wasbary'
	do
		run $lamb $reg $seed
	done

	### ACS Income
	data='acs-income'
	input_size=10
	batch_size=128
	n_classes=1
	nSens=9
	# outfile='results/acsincome_class_res.csv'
	for reg in 'none' 'dp' 'eo' 'demd' 'wasbary'
	do
		run $lamb $reg $seed
	done
}

for seed in 1 2 3
do
	for lamb in 1.0 0.1 10 0.01 100 0.001
	do
		full_loop $seed $lamb &
	done
	wait
done

