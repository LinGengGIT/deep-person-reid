python baseline.py --save-dir log/test \
    --evaluate --resume log/baseline-res50-ilidsvid-eraser/best_model.pth.tar \
	--test-batch 4 --gpu-devices 0 --pool avg
