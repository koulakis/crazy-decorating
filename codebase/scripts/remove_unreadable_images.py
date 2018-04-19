
# coding: utf-8

# # Remove unreadable images

# In[11]:


import os
from skimage.io import imread
import shutil
from multiprocessing import Pool
import json


# In[ ]:


get_ipython().system('mkdir ../data/train_images_readable')
get_ipython().system('mkdir ../data/validation_images_readable')
get_ipython().system('mkdir ../data/test_images_readable')


# In[ ]:


def image_is_readable(image_path):    
    try:
        image = imread(image_path)
        if (len(image.shape) != 3) or (image.shape[2] != 3):
            raise Exception('Invalid image dimensions')
           
        return True
    except Exception as e:
        print(e)
        return False
        
def move_from_a_to_b(image_path, output_directory):
    if image_is_readable(image_path):
        shutil.copy(image_path, output_directory)

def export_list_of_unreadable_images(input_directory, output_directory, filepath):
    unreadable_images = list(set(os.listdir(input_directory)) - set(os.listdir(output_directory)))
    
    with open(filepath, 'w') as path:
        json.dump(unreadable_images, path)

def remove_unreadable(input_directory, output_directory, unreadable_images_filepath, pool_size):
    image_file_names = os.listdir(input_directory)
    image_paths = [os.path.join(input_directory, name) for name in image_file_names]
    
    pool = Pool(pool_size)
    
    for path in image_paths:
        pool.apply_async(move_from_a_to_b, [path, output_directory])
        
    pool.close()
    pool.join()
    
    export_list_of_unreadable_images(input_directory, output_directory, unreadable_images_filepath)


# In[ ]:


remove_unreadable(
    '../data/train_images/', 
    '../data/train_images_readable/', 
    '../data/unreadable_train_images.json', 
    20)


# In[ ]:


remove_unreadable(
    '../data/validation_images/', 
    '../data/validation_images_readable/', 
    '../data/unreadable_validation_images.json', 
    20)


# In[ ]:


remove_unreadable(
    '../data/test_images/', 
    '../data/test_images_readable/', 
    '../data/unreadable_test_images.json', 
    20)


# In[ ]:


get_ipython().system('zip -r ../data/train_images_readable.zip ../data/train_images_readable/')
get_ipython().system('zip -r ../data/validation_images_readable.zip ../data/validation_images_readable/')
get_ipython().system('zip -r ../data/test_images_readable.zip ../data/test_images_readable/')


# In[ ]:


get_ipython().system('aws s3 cp ../data/train_images_readable.zip s3://furniture-kaggle/')
get_ipython().system('aws s3 cp ../data/validation_images_readable.zip s3://furniture-kaggle/')
get_ipython().system('aws s3 cp ../data/test_images_readable.zip s3://furniture-kaggle/')


# In[ ]:


get_ipython().system('aws s3 cp ../data/unreadable_train_images.json s3://furniture-kaggle/')
get_ipython().system('aws s3 cp ../data/unreadable_validation_images.json s3://furniture-kaggle/')
get_ipython().system('aws s3 cp ../data/unreadable_test_images.json s3://furniture-kaggle/')

