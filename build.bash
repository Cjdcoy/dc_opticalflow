    wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb &&\
    sudo dpkg --install cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb && \
    sudo apt-get update && \
    sudo apt-get install cuda -y && \
    git clone https://github.com/Cjdcoy/cudnn5.1.git && \
    tar xvzf cudnn5.1/cudnn-8.0-linux-x64-v5.1.tgz && \
    sudo cp -P cuda/include/cudnn.h /usr/local/cuda/include && \
    sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64 && \
    sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn* && \
    echo export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64 >> ~/.bashrc && \
    echo export CUDA_HOME=/usr/local/cuda >> ~/.bashrc && \
    source ~/.bashrc &&\
    rm -rf cuda  cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb  cudnn5.1 &&\
    \
    sudo apt-get update && \
    sudo   apt-get install -y --no-install-recommends \
	   module-init-tools \
	   build-essential \
	   ca-certificates \
	   wget \
	   git \
	   libatlas-base-dev \
	   libboost-all-dev \
	   libgflags-dev \
	   libgoogle-glog-dev \
	   libhdf5-serial-dev \
	   libleveldb-dev \
	   liblmdb-dev \
	   libopencv-dev \
	   libprotobuf-dev \
	   libsnappy-dev \
	   protobuf-compiler \
	   python-dev \
	   python-numpy \
	   python-scipy \
	   python-protobuf \
	   python-pillow \
	   python-skimage &&\
    \
	   git clone https://github.com/lmb-freiburg/flownet2.git && \
    cp Makefile.config flownet2/ && \
    cd flownet2  && \
    rm -rf .git && \
    cd models && \
    bash download-models.sh && \
    rm flownet2-models.tar.gz && \
    cd .. && \
    sudo make -j`nproc` && \
    sudo make -j`nproc` pycaffe && \
    sudo apt install python3-pip -y && \
    sudo apt install python-pip -y && \
    pip install -r examples/web_demo/requirements.txt && \
    echo "export PYTHONPATH="$PYTHONPATH:~/opticalflow_utilities/flownet2/python/"" >> ~/.bashrc && \
    source ~/.bashrc
