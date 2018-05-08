python baseline.py --save-dir log/test \
    --evaluate --resume log/baseline-res50-ilidsvid/best_model.pth.tar \
	-b 4 --gpu-devices 0 --pool avg
