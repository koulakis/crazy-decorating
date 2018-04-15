
# coding: utf-8

# # Resize images

# In[91]:


from skimage.io import imread
from skimage.transform import resize
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def resize_images(input_dir, output_dir, size):
    filenames = os.listdir(input_dir)
    
    for filename in filenames:
        image_name = filename.split('.')[0]
        try:
            image = imread(os.path.join(input_dir, filename))
            save_path = os.path.join(output_dir, '{}.npy'.format(image_name))
            np.save(
                save_path, 
                resize(image, size))
        except Exception as e:
            print(e)
            print(image_name.split('_')[1])


# In[80]:


widths = []

for filename in paths:
    try:
        image = imread(filename)
        widths.append(image.shape[0])
    except:
        print(filename)


# In[85]:


pd.DataFrame(widths).describe()


# In[83]:


plt.hist(widths, bins=50)
plt.show()


# In[92]:


paths = [os.path.join('../data/validation', filename) for filename in os.listdir('../data/validation')]


# In[67]:


np.load('../data/resized-validation/pic_1.npy').shape


# In[65]:


pics = np.array([np.load(path) for path in paths])


# In[ ]:


pics.shape


# In[61]:


resize_images('../data/validation', '../data/resized-validation', (300, 300))


# In[34]:


min([i.shape[0] for i in images])


# In[36]:


min([i.shape[1] for i in images])


# In[27]:


plt.hist([i.shape[0] for i in images])
plt.show()


# In[28]:


plt.hist([i.shape[1] for i in images])
plt.show()


# In[90]:


plt.clf()

plt.figure(figsize=(10, 100))
for i, image in enumerate(images[:10]):
    plt.subplot(10, 1, i + 1)
    plt.imshow(resize(image, (300, 300)))
plt.show()

