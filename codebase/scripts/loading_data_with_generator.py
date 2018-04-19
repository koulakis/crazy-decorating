
# coding: utf-8

# # Loading data with generator

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
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions, InceptionV3
from keras.models import Sequential

import random


# ## Definition of generator

# In[16]:


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, image_directory, 
                 batch_size=32, dim=(299, 299), n_channels=3,
                 n_classes=128, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.image_directory = image_directory

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            first_time = True
            counter = 0
            while True:
                try:
                    resolved_ID = ID if first_time else random.choice(self.list_IDs)
                    image = imread(os.path.join(self.image_directory, 'pic_{}.png'.format(resolved_ID)))
                    if (len(image.shape) != 3) or (image.shape[2] != 3):
                        raise Exception('Invalid image dimensions')

                    # Store sample   
                    X[i,] = resize(image, [*self.dim, self.n_channels])

                    # Store class
                    y[i] = self.labels[resolved_ID]
                    
                    counter=0
                    break

                except Exception as e:
                    first_time = False
                    
                    counter += 1
                    if counter > 10:
                        print(resolved_ID)
                        print(e)
                        print('Counter: {}'.format(counter))

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


# # Definition of model

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


# # Model training

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train_labels = ((pd
    .read_csv('https://s3-us-west-2.amazonaws.com/furniture-kaggle/train-labels.csv')
    .set_index('image_id')['label_id'] - 1)
    .to_dict())

validation_labels = ((pd
    .read_csv('https://s3-us-west-2.amazonaws.com/furniture-kaggle/validation-labels.csv')
    .set_index('image_id')['label_id'] - 1)
    .to_dict())

train_ids = list(train_labels.keys())
validation_ids = list(validation_labels.keys())


# In[ ]:


number_of_epochs = 20

params = {'dim': (299, 299),
          'batch_size': 32,
          'n_classes': 128,
          'n_channels': 3,
          'shuffle': True}

# Generators
training_generator = DataGenerator(train_ids, train_labels, '../data/train_images/', **params)
validation_generator = DataGenerator(validation_ids, validation_labels, '../data/validation_images/', **params)

transfer_model.fit_generator(
    generator=training_generator,
    validation_data=validation_generator,
    callbacks=callbacks_list,
    verbose=1,
    use_multiprocessing=True,
    max_queue_size=100,
    workers=64)

# Load best weights
transfer_model.load_weights(checkpoints_filepath)


# # Model prediction on test data set

# In[ ]:


test_images_list = load_images('../data/test', (299, 299))


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

