#!/bin/bash

# Set english locale
sudo locale-gen en_CA.UTF-8

# Install common libraries
(cd /home/ubuntu && source /home/ubuntu/anaconda3/bin//activate tensorflow_p36 && pip install sklearn scikit-image)

# Download and unzip datasets
mkdir /home/ubuntu/data
mkdir /home/ubuntu/data/train_images_categorized
mkdir /home/ubuntu/data/validation_images_categorized

(cd /home/ubuntu/ && aws s3 cp --recursive s3://furniture-kaggle/train_images_categorized/ ../data/train_images_categorized/)
(cd /home/ubuntu/ && aws s3 cp --recursive s3://furniture-kaggle/validation_images_categorized/ ../data/validation_images_categorized/)
(cd /home/ubuntu/ && wget https://s3-us-west-2.amazonaws.com/furniture-kaggle/test_images_readable.zip)

(cd /home/ubuntu/ && unzip test_images_readable.zip)
