python st.py --save-dir log/st-res50-ilidsvid \
	-b 32 --gpu-devices 0,1,2,3 --pool qan \
	--optim sgd --lr 0.01 \
	--max-epoch 500 --stepsize 100 --eval-step 50 --print-freq 4
 
 #500,200, 0.0003
