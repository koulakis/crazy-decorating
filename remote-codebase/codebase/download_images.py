import json
import urllib
import multiprocessing
import sys
import pandas as pd
import requests

def read_json(filepath):
    with open(filepath) as data_file:    
        data = json.load(data_file)
    return data

def export_labels(filepath, outputpath):
    data = read_json(filepath)
        
    annotations = data['annotations']
    
    pd.DataFrame(annotations).to_csv(outputpath, index=False)

def download_image(image, directory, timeout):
    output_path = '{}/pic_{}.png'.format(directory, image['image_id'])
    url = image['url'][0]
        
    try: 
        request = requests.get(url, timeout=timeout, stream=True)

        with open(output_path, 'wb') as fh:
            for chunk in request.iter_content(1024 * 1024):
                fh.write(chunk)
    except:
        print(image['image_id'], end=',')
        
    sys.stdout.flush()
    
def download_pictures(filepath, directory, pool_size, timeout):
    data = read_json(filepath)
    
    images = data['images']

    pool = multiprocessing.Pool(pool_size)

    for image in images:
        pool.apply_async(download_image, [image, directory, timeout])

    pool.close()
    pool.join()
    print("Downloaded images for {}".format(filepath))
    

input_json, ouput_dir = sys.argv[1], sys.argv[2]


download_pictures(input_json, output_dir, 500, 1)