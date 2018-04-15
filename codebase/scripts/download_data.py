
# coding: utf-8

# # Download Data

# In[144]:


import json
import urllib
import multiprocessing
import sys
import pandas as pd
import requests
import os


# ## Definitions of functions used to sync images

# In[165]:


def download_image(image_id, url, output_directory, timeout):
    output_path = '{}/pic_{}.png'.format(output_directory, image_id)
        
    try: 
        response = requests.get(url, timeout=timeout, stream=True)
        
        if response.status_code != requests.codes.OK:
            raise 'Request exceeded {} seconds.'.format(timeout)
        
        with open(output_path, 'wb') as fh:
            for chunk in response.iter_content(1024 * 1024):
                fh.write(chunk)
                
    except Exception as e:
        print('{}: {}'.format(image_id, e))

    except:
        print('Unexpected error: {}'.format(sys.exc_info()[0]))
        
    sys.stdout.flush()
    
def download_image_list(image_ids, image_id_to_url, output_directory, pool_size, timeout):
    pool = multiprocessing.Pool(pool_size)
    
    print("Attempting to download {} images in {}".format(len(image_ids), output_directory))
    
    for image_id in image_ids:
        pool.apply_async(download_image, [image_id, image_id_to_url[image_id], output_directory, timeout])

    pool.close()
    pool.join()


# In[186]:


def read_json(filepath):
    with open(filepath) as data_file:    
        data = json.load(data_file)
    return data

def image_id_to_url(filepath):
    data = read_json(filepath)
    
    return dict([
        [link['image_id'], link['url'][0]] 
        for link in data['images']])
    
def images_in_directory(directory):
    return set([
        int(filename.split('_')[1].split('.')[0])
        for filename in os.listdir(directory)])


# In[173]:


def sync_images(id_to_url_filepath, output_directory, pool_size, timeout):
    id_to_url = image_id_to_url(id_to_url_filepath)
    ids = id_to_url.keys()
    
    available_images = images_in_directory(output_directory)

    download_image_list(ids - available_images, id_to_url, output_directory, pool_size, timeout)
    
    return {image_id: id_to_url[image_id]
            for image_id in (ids - available_images)}
    
def iterative_image_download(id_to_url_filepath, output_directory, missing_images_path, pool_size):
    run_sync = lambda t: sync_images(id_to_url_filepath, output_directory, pool_size, t)
    
    unresolved_images = []
    for tmo in (8*[1] + [5]):
        unresolved_images = run_sync(tmo)
    
    with open(missing_images_path, 'w') as file:
        json.dump(unresolved_images, file, sort_keys=True, indent=4)


# ## Syncing images

# In[180]:


id_to_url_filepath = '../data/id-to-url/{}.json'
images_directory = '../data/{}_images/'
missing_images_directory = '../data/missing_images_due_to_bad_download/'
missing_images_filepath = os.path.join(missing_images_directory, 'missing_{}.json')


# In[181]:


for dataset in ['train', 'validation', 'test']:
    get_ipython().system('mkdir -p {images_directory.format(dataset)}')
    
get_ipython().system('mkdir -p {missing_images_directory}')


# In[ ]:


get_ipython().run_cell_magic('time', '', "dataset = 'train'\n\niterative_image_download(\n    id_to_url_filepath.format(dataset),\n    images_directory.format(dataset),\n    missing_images_filepath.format(dataset),\n    100)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "dataset = 'validation'\n\niterative_image_download(\n    id_to_url_filepath.format(dataset),\n    images_directory.format(dataset),\n    missing_images_filepath.format(dataset),\n    100)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "dataset = 'test'\n\niterative_image_download(\n    id_to_url_filepath.format(dataset),\n    images_directory.format(dataset),\n    missing_images_filepath.format(dataset),\n    100)")


# ## Exporting labels

# In[185]:


def export_labels(filepath, outputpath):
    data = read_json(filepath)
        
    annotations = data['annotations']
    
    pd.DataFrame(annotations).to_csv(outputpath, index=False)


# In[3]:


export_labels('../data/train.json', '../data/train-labels.csv')
export_labels('../data/validation.json', '../data/validation-labels.csv')

