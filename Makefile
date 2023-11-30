install_dev_train:
	pip install -r requirements_train.txt
	pip install --upgrade transformers accelerate
	pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex.git
