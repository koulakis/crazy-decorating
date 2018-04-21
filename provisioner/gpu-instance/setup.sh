#!/bin/bash

# Set english locale
sudo locale-gen en_CA.UTF-8

# Install common libraries
(cd /home/ubuntu && source /home/ubuntu/anaconda3/bin//activate tensorflow_p36 && pip install sklearn scikit-image)

# Download and unzip datasets
mkdir /home/ubuntu/data
(cd /home/ubuntu/ && wget https://s3-us-west-2.amazonaws.com/furniture-kaggle/train_images_readable.zip)
(cd /home/ubuntu/ && wget https://s3-us-west-2.amazonaws.com/furniture-kaggle/validation_images_readable.zip)
(cd /home/ubuntu/ && wget https://s3-us-west-2.amazonaws.com/furniture-kaggle/test_images_readable.zip)

(cd /home/ubuntu/ && unzip train_images_readable.zip)
(cd /home/ubuntu/ && unzip validation_images_readable.zip)
(cd /home/ubuntu/ && unzip test_images_readable.zip)
