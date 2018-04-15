import pandas as pd
from skimage.io import imread
from skimage.transform import resize
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions, InceptionV3

train_labels = pd.read_csv('/mnt/66084616-7029-49cb-a245-3f8fad1c7542/Kaggle/furniture/train-lables.csv')

def load_images(input_dir, size, start, lens):
    filenames = os.listdir(input_dir)
    result = []
    for i, filename in enumerate(filenames[start: start+lens]):
        if i%50 == 0:
            print(i,'of',lens)
        image_name = filename.split('.')[0]
        image_id = image_name.split('_')[1]
        try:
            img = image.load_img(os.path.join(input_dir, filename), target_size=size)
            x = image.img_to_array(img)
            x = preprocess_input(x)
            result.append([image_id, x])
        except Exception as e:
            print(image_id)
            pass
            
    return result
	
##### Model Definition 

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

	# can last two layers 
for i in range(311,313):
    transfer_model.layers[i].trainable = True
transfer_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# using checkpoints and early stopping on validation sample to prevent overfitting
# best weight is saved to file_path
file_path="/mnt/66084616-7029-49cb-a245-3f8fad1c7542/Kaggle/furniture/crazy-decorating/checkpoints/weights_base.best.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", patience=5)
callbacks_list = [checkpoint, early] 

##### Training
train_images = load_images('/mnt/66084616-7029-49cb-a245-3f8fad1c7542/Kaggle/furniture/train1', (250, 250), 30000,15000)
train_image_ids = pd.DataFrame([image[0] for image in train_images if image[1].shape[2] == 3], columns=['train_image_id']) 
train_image_labels = pd.concat([
train_image_ids.astype(int).set_index('train_image_id'),
train_labels.set_index('image_id')], join='inner', axis=1)
train_image_labels_np = train_image_labels.reset_index(drop=True).as_matrix()
y_data = train_image_labels_np
epoch = 10
x_data = np.array([image[1] for image in train_images if image[1].shape[2] == 3])

# onehot encoding
y_onehot = np.zeros((y_data.shape[0], num_classes))
for i in range(0,num_classes):
    (y_onehot[:,i:i+1])[y_data==i] = 1

# Load in weights, retrain 	
transfer_model.load_weights(file_path)
transfer_model.fit(x_data, y_onehot, epochs=epoch, validation_split = 0.05, callbacks= callbacks_list, batch_size=32,verbose=1, shuffle = True)

#transfer_model.save(file_path)
##### Predicting test set
test_images = load_images('/mnt/66084616-7029-49cb-a245-3f8fad1c7542/Kaggle/furniture/test', (250, 250),0,5000)
test_predictions = [
    [x[0], transfer_model.predict(x[1].reshape([1, 250, 250, 3]))] 
    for x in test_images
    if (len(x[1].shape) == 3 and x[1].shape[2] == 3)]
test_predictions_dict = dict([[int(x[0]), x[1]] for x in test_predictions])
def predict_or_1(x):
    return test_predictions_dict[x].argmax() if x in test_predictions_dict.keys() else 1
results = pd.DataFrame([[i, predict_or_1(i)] for i in range(1, 12801)], columns=['id', 'predicted'])
results.to_csv('/mnt/66084616-7029-49cb-a245-3f8fad1c7542/Kaggle/furniture/second.csv', index=False)