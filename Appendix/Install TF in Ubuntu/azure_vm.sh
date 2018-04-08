# Cuda setup
CUDA_REPO_PKG=cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
sudo wget -O ${CUDA_REPO_PKG} http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/${CUDA_REPO_PKG} 
sudo dpkg -i ${CUDA_REPO_PKG}
sudo apt-get update
sudo apt-get install cuda-drivers
sudo apt-get install cuda=9.0.176-1
sudo dpkg -i libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb
sudo apt install nvidia-cuda-toolkit

# Upgrade to python3.6
sudo add-apt-repository ppa:jonathonf/python-3.6
sudo apt-get update
sudo apt-get install python3.6
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.5 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 2
echo '====================PLEASE SELECT PYTHON 3.6===================='
echo '====================PLEASE SELECT PYTHON 3.6===================='
echo '====================PLEASE SELECT PYTHON 3.6===================='
sudo update-alternatives --config python3
sudo apt-get install -y python3-pip

# TF & pytorch
sudo pip3 install --upgrade pip
sudo pip3 install tensorflow-gpu==1.6
sudo pip3 install http://download.pytorch.org/whl/cu90/torch-0.3.1-cp36-cp36m-linux_x86_64.whl 
sudo pip3 install torchvision
sudo pip3 install Keras==2.0.7
