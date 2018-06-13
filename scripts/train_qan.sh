python qan.py --save-dir log/debug \
	-b 32 --gpu-devices 4,5,6,7 --pool qan \
	--optim adam --lr 0.001 \
	--max-epoch 500 --stepsize 100 --eval-step 50 --print-freq 4