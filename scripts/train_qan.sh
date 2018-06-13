python qan.py --save-dir log/qan-res50-ilidsvid \
	-b 32 --gpu-devices 0,1,2,3 --pool qan \
	--optim adam --lr 0.0003 \
	--max-epoch 300 --stepsize 200 --eval-step 50 --print-freq 4
