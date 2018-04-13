#!/bin/bash

set -e

# Set english locale
sudo locale-gen en_CA.UTF-8

# Install common libraries
(cd /home/ubuntu && source /home/ubuntu/anaconda3/bin//activate tensorflow_p36 && pip install gensim sklearn)
