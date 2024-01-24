# Packages
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.layers as tfl
import time
import h5py
from datetime import datetime

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation

def newPaths():
    for root,dirs,files in os.walk("./New Folder/val",topdown=True):
        for dir in dirs:
            old = os.path.join(root,dir)
            new = os.path.join(root, '_'.join(dir.split("_")[1:]))
            os.rename(old,new)
            
newPaths()

banner = '\n\n'+'*'*100+'\n\n'
tf.debugging.set_log_device_placement(True)
## Display the gpu hardware in use. 
## GPU will be selected as the default hardware on which to train if found. 
## If not found, an empty list will be returned and CPU (less performant) will be chosen.
gpus = tf.config.list_physical_devices('GPU')
print(f"{banner}\nGPUs: {gpus}\n{banner}")

## Need to figure out how to run model with this cache optimization,
## and if it actually improves training performance.
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = (224,224)
def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

def data_augmenter():
        '''
        Create a Sequential model composed of 2 layers
        Returns:
            tf.keras.Sequential
        '''
        data_augmentation = tf.keras.Sequential()
        data_augmentation.add(RandomFlip('horizontal'))
        data_augmentation.add(RandomRotation(factor = 1, seed = 1, fill_mode='nearest'))
        
        return data_augmentation

def fungus_model(image_shape=IMG_SIZE, data_augmentation=data_augmenter()):
  ''' Define a tf.keras model for binary classification out of the MobileNetV2 model
  Arguments:
      image_shape -- Image width and height
      data_augmentation -- data augmentation function
  Returns:
  Returns:
      tf.keras.model
  '''


  input_shape = image_shape + (3,)
  print(input_shape)
  base_model = tf.keras.applications.MobileNetV3(input_shape=input_shape,
                                              include_top=False, # <== Important!!!!
                                              weights='imagenet') # From imageNet

  # freeze the base model by making it non trainable
  base_model.trainable = False 

  # create the input layer (Same as the imageNetv2 input size)
  inputs = tf.keras.Input(shape=input_shape) 

  # apply data augmentation to the inputs
  x = data_augmentation(inputs)

  # data preprocessing using the same weights the model was trained on
  x = preprocess_input(x) 

  # set training to False to avoid keeping track of statistics in the batch norm layer
  x = base_model(x, training=False) 

  # add the new Binary classification layers
  # use global avg pooling to summarize the info in each channel
  x = tfl.GlobalAveragePooling2D()(x) 
  # include dropout with probability of 0.2 to avoid overfitting
  x = tfl.Dropout(0.2)(x)
      
  # use a prediction layer with one neuron (as a binary classifier only needs one)
  outputs = tfl.Dense(224, activation='softmax')(x)
      
  model = tf.keras.Model(inputs, outputs)

  return model
## Creating training and validation sets for the model to train on.
## return of the called functs is an array of tensors.
train_directory = "./New Folder/val/"
val_directory = "./New Folder/val/"
print("building training set")
start = time.time()
train_dataset = image_dataset_from_directory( train_directory,
                                              labels = 'inferred',
                                              label_mode = 'int',
                                              shuffle=True,
                                              image_size=IMG_SIZE,
                                              seed=42,
                                              validation_split = 0.2,
                                              subset = 'training',
                                              crop_to_aspect_ratio = True)
print(f"{banner}training set built in {time.time()-start} seconds{banner}")
print(f"{banner}building validation set{banner}")
start = time.time()
validation_dataset = image_dataset_from_directory(  val_directory,
                                                    labels = 'inferred',
                                                    label_mode = 'int',
                                                    shuffle=True,
                                                    image_size=IMG_SIZE,
                                                    seed=42,
                                                    validation_split = 0.2,
                                                    subset = 'validation',
                                                    crop_to_aspect_ratio = True)
print(f"{banner}validation set built in {time.time()-start} seconds{banner}")
# class_names = train_dataset.class_names
image_batch, label_batch = next(iter(train_dataset))
print(label_batch)
# input("PRESS ENTER TO CONTINUE")
## This function should cache the tensors from the dataset and optimize model training performance.
## However, I'm not sure how to make the model run with this caching as it adds
## dimensionality to the dataset. "Optimization" commented out for now.
# train_dataset = configure_for_performance(train_dataset)
# validation_dataset = configure_for_performance(validation_dataset)


## prefetch data. What's it do? let's see...
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
print(train_dataset)

## Take a look at how an image from the training set has been augmented with simple transformations:

start = time.time()
data_augmentation = data_augmenter()
print(f"{banner}Preprocessing input...{banner}")
start = time.time()
preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
print(f"{banner}preprocessing input complete in {time.time()-start} seconds{banner}")

print(f"{banner}establishing base model{banner}")
IMG_SHAPE = IMG_SIZE + (3,)
start = time.time()
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint( filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=1)
base_model = tf.keras.applications.MobileNetV3Large(  input_shape=IMG_SHAPE,
                                                      include_top=True,
                                                      weights='imagenet')
print(f"{banner}establishing base model complete in {time.time()-start} seconds{banner}")
print(f"{banner}BASE MODEL:")
base_model.summary()
print(banner)

# input("press enter to continue")

# print(feature_batch.shape)

# print(label_batch)

# arr = np.random.randn(100)
# with h5py.File('model.weights.h5', 'w') as f:
#   dset = f.create_dataset("default", data=arr)
#   data = f['default']
#   print(dset)
#   print(min(data))
#   print(max(data))
#   print(data[:15])
#   for key in f.keys():
#    print(key)
# input("continue?")
fungal_model = tf.keras.models.load_model('MobileNetV3_fungus.keras')
fungal_model.compile( optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss = 'sparse_categorical_crossentropy',
                      metrics = ['accuracy'],
                      run_eagerly = False)
print(fungal_model.summary())
# input("continue?")
# print(fungal_model.predict(validation_dataset))
# fungal_model.fit( train_dataset, 
#                   epochs=1,
#                   validation_data=validation_dataset,
#                   callbacks=[cp_callback],
#                   verbose = 1)
image_batch, label_batch = next(iter(train_dataset))
preds = []

image_var = tf.Variable(preprocess_input(image_batch))
preds =fungal_model.predict(image_var)
print(preds)
for pred in preds:
    print(pred,"\n")
    print(np.argmax(pred))
