# # Transfer Learning with MobileNetV2
# GOALS:
# - Create a dataset from a directory
# - Preprocess and augment data using the Sequential API
# - Adapt a pretrained model to new data and train a classifier using the Functional API and MobileNet
# - Fine-tune a classifier's final layers to improve accuracy


# Packages
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.layers as tfl
import time
from datetime import datetime

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation

banner = '\n\n****************************************************************\n\n'
tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_physical_devices('GPU')
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE
def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds
  
print(gpus)
data_augmentation = tf.keras.Sequential([
  tfl.RandomFlip("horizontal_and_vertical"),
  tfl.RandomRotation(0.2),
])
# Create the Dataset and Split it into Training and Validation Sets
with tf.device('/GPU:0'):
    BATCH_SIZE = 32
    IMG_SIZE = (224,224)
    train_directory = "./New Folder/"
    val_directory = "./New Folder/"
    print("building training set")
    start = time.time()
    train_dataset = image_dataset_from_directory(train_directory,
                                                labels = 'inferred',
                                                label_mode = 'int',
                                                shuffle=True,
                                                image_size=IMG_SIZE,
                                                seed=42,
                                                validation_split = 0.2,
                                                subset = 'training',
                                                crop_to_aspect_ratio = True)
    #train_dataset = configure_for_performance(train_dataset)
    print(f"{banner}building training set complete in {time.time()-start} seconds")
    print("building validation set")
    start = time.time()
    validation_dataset = image_dataset_from_directory(val_directory,
                                                labels = 'inferred',
                                                label_mode = 'int',
                                                shuffle=True,
                                                image_size=IMG_SIZE,
                                                seed=42,
                                                validation_split = 0.2,
                                                subset = 'validation',
                                                crop_to_aspect_ratio = True)
    #validation_dataset = configure_for_performance(validation_dataset)
    print(f"{banner}building validation set complete in {time.time()-start} seconds")


# Now let's take a look at some of the images from the training set: 
print(train_dataset)
#class_names = train_dataset.class_names
# print(class_names)
# plt.figure(figsize=(10, 10))
# for images, labels in train_dataset.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[labels[i]])
#         plt.axis("off")
# show sample of images from dataset
# plt.show()



# ## 2 - Preprocess and Augment Training Data
with tf.device('/GPU:0'):
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

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

    # # Take a look at how an image from the training set has been augmented with simple transformations:
    print("creating data_augmentation")
    start = time.time()
    data_augmentation = data_augmenter()
    print(f"{banner}creating data_augmentation in {time.time()-start} seconds")

    # for image, _ in train_dataset.take(1):
    #     plt.figure(figsize=(10, 10))
    #     first_image = image[0]
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
    #         plt.imshow(augmented_image[0] / 255)
    #         plt.axis('off')
    # plt.show()

    # # Next, you'll apply your first tool from the MobileNet application in TensorFlow, to normalize your input. Since you're using a pre-trained model that was trained on the normalization values [-1,1], it's best practice to reuse that standard with tf.keras.applications.mobilenet_v2.preprocess_input.
    print("preprocessing input...")
    start = time.time()
    preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
    print(f"{banner}preprocessing input complete in {time.time()-start} seconds")

    # # ## 3 - Using MobileNetV2 for Transfer Learning 



    IMG_SHAPE = IMG_SIZE + (3,)
    print("establishing base model")
    start = time.time()
    base_model = tf.keras.applications.MobileNetV3Large(input_shape=IMG_SHAPE,
                                                include_top=True,
                                                weights='imagenet')
    print(f"{banner}establishing base model complete in {time.time()-start} seconds")


    # # Print the model summary below to see all the model's layers, the shapes of their outputs, and the total number of parameters, trainable and non-trainable. 

    base_model.summary()


    # # Note the last 2 layers here. They are the so called top layers, and they are responsible of the classification in the model


    nb_layers = len(base_model.layers)
    print(base_model.layers[nb_layers - 2].name)
    print(base_model.layers[nb_layers - 1].name)


    # # Notice some of the layers in the summary like `Conv2D` and `DepthwiseConv2D` and how they follow the progression of expansion to depthwise convolution to projection. In combination with BatchNormalization and ReLU, these make up the bottleneck layers mentioned earlier.


    image_batch, label_batch = next(iter(train_dataset))
    feature_batch = base_model(image_batch)
    print(feature_batch.shape)

    # #Shows the different label probabilities in one tensor 
    label_batch


    base_model.trainable = False
    image_var = tf.Variable(preprocess_input(image_batch))
    pred = base_model(image_var)

    print(tf.keras.applications.mobilenet_v2.decode_predictions(pred.numpy(), top=2))


    # 3.2 - Layer Freezing with the Functional API

    def alpaca_model(image_shape=IMG_SIZE, data_augmentation=data_augmenter()):
        ''' Define a tf.keras model for binary classification out of the MobileNetV2 model
        Arguments:
            image_shape -- Image width and height
            data_augmentation -- data augmentation function
        Returns:
        Returns:
            tf.keras.model
        '''
        
        
        input_shape = image_shape + (3,)
            
        base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
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



    model2 = alpaca_model(IMG_SIZE, data_augmentation)


    base_learning_rate = 0.001
    print("compiling model2")
    model2.compile( optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    print(f"{banner}compiling model2 complete in {time.time()-start} seconds")


    # # In[ ]:


    initial_epochs = 2
    print("fitting model2 to new output classes")
    history = model2.fit(   train_dataset, 
                            validation_data=validation_dataset,
                            epochs=initial_epochs, batch_size = 32, 
                            verbose=1)
    print(f"{banner}fitting model2 to new output classes complete in {time.time()-start} seconds")


    # # Plot the training and validation accuracy:

    # # In[ ]:


    acc = [0.] + history.history['accuracy']
    val_acc = [0.] + history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,2])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig(f"{datetime.now()}classes_reset.png")


    # print(class_names)

    #  3.3 - Fine-tuning the Model

    base_model = model2
    base_model.trainable = True
    # A look to see how many layers are in the base model
    print("Number of layers in the base model: ", len(base_model.layers))

    # Fine-tune from this layer onwards
    fine_tune_at = 30

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[0:fine_tune_at]:
        layer.trainable = False
        
    # Define a BinaryCrossentropy loss function. Use from_logits=True
    loss_function='sparse_categorical_crossentropy'
    # Define an Adam optimizer with a learning rate of 0.1 * base_learning_rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=.001)
    # Use accuracy as evaluation metric
    metrics=['accuracy']

    model2.compile(loss=loss_function,
                optimizer = optimizer,
                metrics=metrics,
                run_eagerly = False)

    fine_tune_epochs = 20
    total_epochs =  initial_epochs + fine_tune_epochs
    print("fine tuning model2 ")
    history_fine = model2.fit(train_dataset,
                            epochs=total_epochs,
                            initial_epoch=history.epoch[-1],
                            validation_data=validation_dataset,
                            batch_size = 32,
                            verbose=1)
    print(f"{banner}fine tuning model2 complete in {time.time()-start} seconds")
    model2.save(f'MobileNetV3_fungus.keras{datetime.datetime.now()}')
    




    acc += history_fine.history['accuracy']
    val_acc += history_fine.history['val_accuracy']

    loss += history_fine.history['loss']
    val_loss += history_fine.history['val_loss']


    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.ylim([0, 1])
    plt.plot([initial_epochs-1,initial_epochs-1],
            plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylim([0, 2])
    plt.plot([initial_epochs-1,initial_epochs-1],
            plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig(f"{datetime.now()}finaltrain.png")

    print(history_fine.history.keys())

    # # That's awesome! 
