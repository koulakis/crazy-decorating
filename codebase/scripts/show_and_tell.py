
# coding: utf-8

# In[10]:


import cv2
import time

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    
    if frame:
        cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(1)

        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1  
            # find prediction 
            img = image.load_img(img_name, target_size=(250, 250))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            preds = transfer_model.predict(x)
            pred_class = preds.argmax()

cam.release()

cv2.destroyAllWindows()


# In[31]:


categories = { 
    "chair": [1, 3, 4, 6, 15, 22, 23, 26, 29, 43, 55, 58, 63, 104],
    "table": [8, 13, 14, 19, 59, 61, 80, 83, 90, 96, 103, 113, 126, 128],
    "accessories": [2, 9, 12, 41, 50, 53, 54, 56, 60, 68, 69, 71, 88, 98, 111, 116, 123],
    "kitchen_items": [5, 11, 16, 20, 27, 33, 44, 46, 49, 57, 64, 72, 73, 74, 79, 82, 87, 95, 100, 102, 105, 106, 107, 108, 109, 110, 112, 114, 118, 125, 127],
    "light": [30, 37, 51, 84, 89, 93, 97],
    "bed": [10, 28, 35, 38, 70, 92, 117],
    "couch": [17, 18, 21, 45, 48, 86, 91, 94, 119, 120],
    "electronics": [7, 36, 42, 52, 77, 78, 81, 115, 124],
    "shelves": [24, 25, 31, 32, 34, 40, 47, 62, 65, 66, 67, 75, 85, 122, 121],
    "carpet": [99],
    "other": [39, 76, 101]
}


# In[32]:


import functools
    
len(functools.reduce(lambda x, y: x + y, categories.values()))


# In[33]:


import pandas as pd
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import os

import keras
from keras.models import Model
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions, InceptionV3
from keras.models import Sequential

import random


# In[34]:


num_classes = 128

model = InceptionV3(weights='imagenet')

intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[311].output)

x = intermediate_layer_model.output
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

transfer_model = Model(inputs=intermediate_layer_model.input, outputs=predictions)

# train last cluster and dense layer
for layer in transfer_model.layers:
    layer.trainable = False

# can train more 
for i in range(311,313):
    transfer_model.layers[i].trainable = True

transfer_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# using checkpoints and early stopping on validation sample to prevent overfitting
# best weight is saved to file_path
checkpoints_filepath="../checkpoints/weights_base.best.hdf5"
checkpoint = ModelCheckpoint(checkpoints_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", patience=2)
callbacks_list = [checkpoint, early] 


# In[35]:


transfer_model.load_weights('../checkpoints/weights_base_2.best.hdf5')


# In[37]:


from skimage.io import imread
from skimage.transform import resize
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_images(input_dir, size):
    filenames = os.listdir(input_dir)
    
    result = []
    
    for filename in filenames:
        image_name = filename.split('.')[0]
        image_id = image_name.split('_')[1]
        try:
            image = imread(os.path.join(input_dir, filename))
            if (len(image.shape) != 3) or (image.shape[2] != 3):
                raise Exception('Bad image dimensions')
            result.append([image_id, resize(image, size)])
        except Exception as e:
            pass
            print(e)
            print(image_name.split('_')[1])
            
    return result


# In[ ]:


val_images = load_images('../data/validation/', (299, 299))


# In[40]:


online_images = load_images('../online_images/', (299, 299))


# In[41]:


online_images


# In[55]:


transfer_model.predict(online_images[0][1].reshape([1, 299, 299, 3])).argmax() + 1

