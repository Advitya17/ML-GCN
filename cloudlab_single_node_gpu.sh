#!/bin/bash

########################## Setup Instructions ###############################
# 1. In CloudLab select the select-hardware profile
# 2. Select hardware type to c240g5
# 3. Login to the node and disable nouveau (https://linuxconfig.org/how-to-disable-nouveau-nvidia-driver-on-ubuntu-18-04-bionic-beaver-linux)
# 4. Then run this script in user's home dir
# 4. (Optional) add any other depdencies (e.g., python libs) to this script
#############################################################################

# For mounting repo to a different workspace
# cd $HOME
# mkdir workspace
# cd workspace

# sudo mkfs.ext4 /dev/sdb
# sudo mount /dev/sdb $HOME/workspace

# sudo chown $USER:orion-PG0 -R $HOME/workspace
# git clone https://github.com/Advitya17/ML-GCN 
# --------------------------------------

cd $HOME
mkdir code
cd code

sudo mkfs.ext3 /dev/sda4
sudo mount /dev/sda4 $HOME/code

sudo chown $USER:orion-PG0 -R $HOME/code


sudo apt-get update

# build tools
sudo apt-get install build-essential cmake unzip pkg-config -y

#optimization libraries
sudo apt-get install libopenblas-dev libatlas-base-dev liblapack-dev gfortran -y

# python
sudo apt-get install python3-dev python3-tk python-imaging-tk -y
sudo apt-get install libgtk-3-dev -y
sudo apt-get install python3-pip -y
sudo python3 -m pip install --upgrade pip

# cuda 10.1
wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run
sudo sh cuda_10.1.243_418.87.00_linux.run --toolkit --driver --toolkitpath=~/code/cuda --silent

echo 'export PATH=$HOME/code/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/vista/cuda/lib64:$HOME/cuda/extras/CUPTI/lib64' >> ~/.bashrc
echo 'export CUDA_HOME=$HOME/code/cuda' >> ~/.bashrc
source ~/.bashrc

# cudnn
mkdir cudnn
cd cudnn
cp /proj/orion-PG0/cudnn-10.1-linux-x64-v7.6.5.32.tgz ./
tar -xzvf cudnn-10.1-linux-x64-v7.6.5.32.tgz
sudo cp cuda/include/cudnn.h $HOME/code/cuda/include
sudo cp cuda/lib64/libcudnn* $HOME/code/cuda/lib64
sudo chmod a+r $HOME/code/cuda/include/cudnn.h $HOME/code/cuda/lib64/libcudnn*
cd ..
sudo rm -rf cudnn

# java
sudo apt-get install -y openjdk-8-jdk-headless -qq > /dev/null
echo 'export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64' >> ~/.bashrc
echo 'export PATH=$PATH:$JAVA_HOME/bin' >> ~/.bashrc
source ~/.bashrc


# python libraries
sudo python3 -m pip install tensorflow-gpu==2.2.0
sudo python3 -m pip install pandas
sudo python3 -m pip install pyarrow
sudo python3 -m pip install pyspark==2.4
sudo python3 -m pip install jupyter
sudo python3 -m pip install scikit-learn
# sudo python3 -m pip install torch==0.3.1 torchnet torchvision==0.2.0 tqdm

sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 1
