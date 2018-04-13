#!/bin/bash

set -e

# Set english locale
sudo locale-gen en_CA.UTF-8

# Installs anaconda
wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
chmod +x Anaconda3-5.0.1-Linux-x86_64.sh
./Anaconda3-5.0.1-Linux-x86_64.sh -b

# Exports anaconda bin to path
echo 'export PATH=~/anaconda3/bin:$PATH' >> ~/.bashrc
export PATH=~/anaconda3/bin:$PATH

# Install common libraries
conda install -y pandas scikit-learn scipy
