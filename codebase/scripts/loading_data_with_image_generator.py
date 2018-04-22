
# coding: utf-8

# # Loading data with image generator

# In[15]:


import pandas as pd
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import os

import keras
from keras.models import Model
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input, decode_predictions, InceptionV3
from keras.models import Sequential

import random


# ## Definition of model

# In[17]:


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


# ## Definition of generator

# In[ ]:


batch_size = 32
target_size = (299, 299)

train_images_directory = '../data/train_images_categorized/'
validation_images_directory = '../data/validation_images_categorized/'

train_datagen = ImageDataGenerator()
validation_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_images_directory, 
    target_size=target_size, 
    batch_size=batch_size) 

validation_generator = validation_datagen.flow_from_directory(
    validation_images_directory,
    target_size=target_size,
    batch_size=batch_size)


# ## Model training

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


number_of_epochs = 20

transfer_model.fit_generator(
    generator=train_generator,
    validation_data=validation_generator,
    callbacks=callbacks_list,
    verbose=1,
    epochs=number_of_epochs,
    use_multiprocessing=True,
    max_queue_size=300,
    workers=100)


# In[ ]:


# Load best weights
transfer_model.load_weights(checkpoints_filepath)


# ## Model prediction on test data set

# In[ ]:


def load_images(input_dir, size):
    return [
        [filename.split('.')[0].split('_')[1],
         resize(
             imread(os.path.join(input_dir, filename)), 
             size)] 
        for filename in os.listdir(input_dir)]


# In[ ]:


test_images_list = load_images('../data/test_images_readable/', (299, 299))


# In[ ]:


test_image_array = np.array(
    [image[1] 
     for image in test_images_list
     if image[1].shape[2] == 3])


# In[ ]:


test_predictions = [
    [x[0], transfer_model.predict(x[1].reshape([1, 299, 299, 3]))] 
    for x in test_images_list 
    if (len(x[1].shape) == 3 and x[1].shape[2] == 3)]


# In[ ]:


test_predictions_dict = dict([[int(x[0]), x[1]] for x in test_predictions])


# In[ ]:


def predict_or_1(x):
    return test_predictions_dict[x].argmax() + 1 if x in test_predictions_dict.keys() else 1


# In[ ]:


results = pd.DataFrame([[i, predict_or_1(i)] for i in range(1, 12801)], columns=['id', 'predicted'])


# In[ ]:


results.to_csv('first-submission.csv', index=False)

