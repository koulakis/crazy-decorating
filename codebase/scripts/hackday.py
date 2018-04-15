
# coding: utf-8

# In[4]:


from keras.models import Model
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions, InceptionV3

import numpy as np
import pandas as pd


# In[ ]:


# load in preprocessed data


# In[3]:


model = InceptionV3(weights='imagenet')

intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[311].output)

x = intermediate_layer_model.output
x = Dense(1024, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)

transfer_model = Model(inputs=intermediate_layer_model.input, outputs=predictions)

# train last cluster and dense layer
for layer in transfer_model.layers:
    layer.trainable = False

# can train more 
for i in range(311,313):
    transfer_model.layers[i].trainable = True

transfer_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

################### training last X layers ###################
epoch = 20
num_classes = 128
# onehot encoding
y_onehot = np.zeros((y_data.shape[0], num_classes))
for i in range(0,num_classes):
    (y_onehot[:,i:i+1])[y_data==i] = 1

# using checkpoints and early stopping on validation sample to prevent overfitting
# best weight is saved to file_path
file_path=".../weights_base.best.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", patience=2)
callbacks_list = [checkpoint, early] 

transfer_model.fit(x_data, y_onehot, epochs=epoch, validation_split=0.1, batch_size=32, callbacks=callbacks_list)

# load best weights
transfer_model.load_weights(file_path)

# need prediction


# In[ ]:




