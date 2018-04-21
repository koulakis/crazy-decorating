
# coding: utf-8

# # Resize images

# In[1]:


import os
from skimage.io import imread
from skimage.transform import resize
from multiprocessing import Pool


# In[ ]:


get_ipython().system('mkdir ../data/train_images_resized')
get_ipython().system('mkdir ../data/validation_images_resized')
get_ipython().system('mkdir ../data/test_images_resized')


# In[ ]:


def resize_and_export(filename, output_directory, size):
    image = imread(os.path.join(input_dir, filename))
    save_path = os.path.join(output_directory, '{}.npy'.format(filename))
    np.save(
        save_path,
        resize(image, size))

def resize_images(input_directory, output_directory, size, pool_size):
    image_file_names = os.listdir(input_directory)
    
    pool = Pool(pool_size)
    
    for filename in image_file_names:
        pool.apply_async(resize_and_export, [filename, input_directory, output_directory, size])
        
    pool.close()
    pool.join()


# In[ ]:


resize_images(
    '../data/train_images_readable/', 
    '../data/train_images_resized/', 
    (299, 299), 
    20)


# In[ ]:


resize_images(
    '../data/validation_images_readable/', 
    '../data/validation_images_resized/', 
    (299, 299), 
    20)


# In[ ]:


resize_images(
    '../data/test_images_readable/', 
    '../data/test_images_resized/', 
    (299, 299), 
    20)

