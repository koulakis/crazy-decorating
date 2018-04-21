
# coding: utf-8

# # Move images to category directories
# So that keras can autmatically import them.

# In[20]:


import os
from multiprocessing import Pool
import pandas as pd
from shutil import copyfile


# In[85]:


def categorize_image(image_id, input_directory, output_directory, label): 
    copyfile(
        os.path.join(input_directory, 'pic_{}.png'.format(image_id)),
        os.path.join(output_directory, str(label + 1), 'pic_{}.png'.format(image_id)))

def categorize_images(input_directory, output_directory, labels, pool_size):
    image_filenames = os.listdir(input_directory)
    image_ids = [int(filename.split('.')[0].split('_')[1]) for filename in image_filenames]
    
    pool = Pool(pool_size)
    
    for image_id in image_ids:
        pool.apply_async(categorize_image, [image_id, input_directory, output_directory, labels[image_id]])
        
    pool.close()
    pool.join()


# In[ ]:


get_ipython().system('mkdir ../data/train_images_categorized')
get_ipython().system('mkdir ../data/validation_images_categorized')


# In[ ]:


for i in range(1, 129):
    get_ipython().system('mkdir ../data/train_images_categorized/{i}')


# In[17]:


for i in range(1, 129):
    get_ipython().system('mkdir ../data/validation_images_categorized/{i}')


# In[ ]:


train_images_directory = '../data/train_images_readable/'
validation_images_directory = '../data/validation_images_readable/'

train_labels = ((pd
    .read_csv('https://s3-us-west-2.amazonaws.com/furniture-kaggle/train-labels.csv')
    .set_index('image_id')['label_id'] - 1)
    .to_dict())

validation_labels = ((pd
    .read_csv('https://s3-us-west-2.amazonaws.com/furniture-kaggle/validation-labels.csv')
    .set_index('image_id')['label_id'] - 1)
    .to_dict())

train_filenames = os.listdir(train_images_directory)
train_ids = [int(filename.split('.')[0].split('_')[1]) for filename in train_filenames]

validation_filenames = os.listdir(validation_images_directory)
validation_ids = [int(filename.split('.')[0].split('_')[1]) for filename in validation_filenames]


# In[92]:


categorize_images(
    train_images_directory, 
    '../data/train_images_categorized/', 
    train_labels, 
    200)


# In[86]:


categorize_images(
    validation_images_directory, 
    '../data/validation_images_categorized/', 
    validation_labels, 
    200)


# In[ ]:


get_ipython().system('aws s3 cp --recursive ../data/train_images_categorized/ s3://furniture-kaggle/train_images_categorized/')
get_ipython().system('aws s3 cp --recursive ../data/validation_images_categorized/ s3://furniture-kaggle/validation_images_categorized/')

