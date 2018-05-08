python main.py --save-dir log/test \
    --evaluate --resume log/qan-res50-ilidsvid/best_model.pth.tar \
	-b 4 --gpu-devices 0 --pool qan
