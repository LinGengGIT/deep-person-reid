python baseline.py --save-dir log/baseline-res50-ilidsvid-eraser \
	-b 32 --gpu-devices 4,5,7 \
	--optim adam --lr 0.0003 \
	--max-epoch 500 --stepsize 200 --eval-step 50 --print-freq 4