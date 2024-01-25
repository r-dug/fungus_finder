# # Transfer Learning with MobileNetV2
# GOALS:
# - Create a dataset from a directory
# - Preprocess and augment data using the Sequential API
# - Adapt a pretrained model to new data and train a classifier using the Functional
#    API and MobileNet
# - Fine-tune a classifier's final layers to improve accuracy


# Packages
import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf
import tensorflow.keras.layers as tfl
import time
from datetime import datetime

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation
rank = {'1': 'Kingdom',
'2': 'Phylum',
'3': 'Class',
'4': 'Order',
'5': 'Family',
'6': 'Genus',
'7': 'Species'}
taxonomic_rank = rank[sys.argv[1]]
print(taxonomic_rank)
banner = '\n\n'+'*'*100+'\n\n'
tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_physical_devices('GPU')
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32
IMG_SIZE = (224,224)
train_directory = "./../DATASETS/train/"
val_directory = "./../DATASETS/train/"
print(f"{banner}GPUs: {gpus} {banner}")
input("enter")

def plot_performance(phase, acc, val_acc, loss, val_loss):
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
    plt.title(f'{taxonomic_rank}\nTraining and Validation Performance\nfor {phase}')
    plt.xlabel('epoch')
    plt.savefig(f"./../PERFORMANCE/{datetime.now()}_{phase}{taxonomic_rank}.png")

def data_augmenter():
    '''
    Create a Sequential model composed of 2 layers
    Returns:
        tf.keras.Sequential
    '''
    data_augmentation = tf.keras.Sequential()
    data_augmentation.add(tfl.RandomFlip('horizontal_and_vertical'))

    data_augmentation.add(tfl.RandomRotation(   factor = 0.75, 
                                                fill_mode='nearest'))

    data_augmentation.add(tfl.RandomTranslation(height_factor = 0.1,
                                                width_factor = 0.1,
                                                fill_mode = "nearest",
                                                interpolation = "nearest"))
    
    # data_augmentation.add(tfl.RandomHeight( factor = 0.1))

    # data_augmentation.add(tfl.RandomWidth(  factor = 0.1))

    # data_augmentation.add(tfl.RandomZoom(   height_factor = (0.0, 0.1),
    #                                         width_factor=(0.0, 0.1),
    #                                         fill_mode='nearest',
    #                                         interpolation='bilinear'))

    data_augmentation.add(tfl.RandomBrightness( factor = 0.2, 
                                                value_range=(0, 255), 
                                                seed=None))
    
    data_augmentation.add(tfl.RandomContrast(   factor = 0.2))
    return data_augmentation
# # instantiates augmenter layers
data_augmentation = data_augmenter()

def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

def fungus_model(image_shape=IMG_SIZE, data_augmentation=data_augmenter()):
    ''' Define a tf.keras model for binary classification out of the mobilenet_v3Large
      model
    Arguments:
        image_shape -- Image width and height
        data_augmentation -- data augmentation function
    Returns:
    Returns:
        tf.keras.model
    '''
    
    
    input_shape = image_shape + (3,)
        
    base_model = tf.keras.applications.MobileNetV3Large(input_shape=input_shape,
                                                        include_top=False, # <== Important!!!!
                                                        weights='imagenet',
                                                        minimalistic=True) # From imageNet
    
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
    outputs = tfl.Dense(3, activation='softmax')(x)
        
    model = tf.keras.Model(inputs, outputs)
    
    return model

# Create the Dataset and Split it into Training and Validation Sets
with tf.device('/GPU:0'):
    print("building training set")
    start = time.time()
    train_dataset = image_dataset_from_directory(   train_directory,
                                                    labels = 'inferred',
                                                    label_mode = 'int',
                                                    shuffle=True,
                                                    image_size=IMG_SIZE,
                                                    seed=42,
                                                    validation_split = 0.2,
                                                    subset = 'training',
                                                    crop_to_aspect_ratio = True)
    #train_dataset = configure_for_performance(train_dataset)
    print(f"{banner}building training set complete in {time.time()-start} seconds{banner}")
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
    #validation_dataset = configure_for_performance(validation_dataset)
    print(f"{banner}building validation set complete in {time.time()-start} seconds{banner}")

    # # Shows the tensor iterator. shows the class names
    # print(train_dataset)
    class_names = train_dataset.class_names
    # for class_name in class_names:
    #     print(class_name)
    # print(banner)
    input("Press enter to continue")

    # # Option to look at some of the images from the training set: 
    # plt.figure(figsize=(10, 10))
    # for images, labels in train_dataset.take(1):
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.imshow(images[i].numpy().astype("uint8"))
    #         plt.title(class_names[labels[i]])
    #         plt.axis("off")
    # plt.show()

    # ## 2 - Preprocess and Augment Training Data
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

    # # Option to look at how an image from the training set has been augmented with
    # # transformations:

    # for image, label in train_dataset.take(1):
    #     plt.figure(figsize=(10, 10))
    #     first_image = image[0]
    #     for j in range (9):
    #         for i in range(9):
    #             ax = plt.subplot(3, 3, i + 1)
    #             augmented_image = data_augmentation(tf.expand_dims(image[j], 0))
    #             plt.title(class_names[label[j]])
    #             plt.imshow(augmented_image[0] / 255)
    #             plt.axis('off')
    #         plt.show()

    # # Use TensorFlow, to normalize your input. Since using a pre-trained model
    # # that was trained on the normalization values [-1,1], it's best practice to
    # # reuse that standard with tf.keras.applications.mobilenet_v3.preprocess_input.
    print(f"{banner}preprocessing input...{banner}")
    start = time.time()
    preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
    print(f"{banner}preprocessing input complete in {time.time()-start} seconds{banner}")

    # # Using mobilenet_v3Large for Transfer Learning 
    IMG_SHAPE = IMG_SIZE + (3,)
    print(f"{banner}establishing base model{banner}")
    start = time.time()
    base_model = tf.keras.applications.MobileNetV3Large(    input_shape=IMG_SHAPE,
                                                            include_top=True,
                                                            weights='imagenet')
    print(f"{banner} base model established in {time.time()-start} seconds{banner}")


    # # Save the model summary below to see all the model's layers, the shapes of
    # # their outputs, and the total number of parameters, trainable and 
    # # non-trainable. 
    with open(f"./../MODEL SUMMARIES.{datetime.now()}base_model_summary.txt", 'w', encoding='utf-8') as file:
        print(base_model.summary(), file = file)


    # # Notice some of the layers in the summary like `Conv2D` and `DepthwiseConv2D`
    # # and how they follow the progression of expansion to depthwise convolution to
    # # projection. In combination with BatchNormalization and ReLU, these make up 
    # # the bottleneck layers.
    image_batch, label_batch = next(iter(train_dataset))
    feature_batch = base_model(image_batch)

    # # Shows the different labeled predictions based on mobilenet labels 
    label_batch
    base_model.trainable = False
    image_var = tf.Variable(preprocess_input(image_batch))
    pred = base_model(image_var)
    print(tf.keras.applications.mobilenet_v3.decode_predictions(    pred.numpy(), 
                                                                    top=2))
    print(f"{banner}Predictions with labels from mobilenetv3 {banner}")
    # # Using the function defined earlier to create a new model instance.
    model2 = fungus_model(IMG_SIZE, data_augmentation)


    base_learning_rate = 0.001
    print("compiling model2")
    model2.compile( optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    print(f"{banner}compiling model2 complete in {time.time()-start} seconds")

    # # Here we are only renaming the classes and assigning new probabilities to them.
    # # I.E. creating a new softmax layer
    initial_epochs = 2
    print("fitting model2 to new output classes")
    history = model2.fit(   train_dataset, 
                            validation_data=validation_dataset,
                            epochs=initial_epochs, batch_size = 32, 
                            verbose=1)
    print(f"{banner}fitting model2 to new output classes complete in {time.time()-start} seconds")


    # # Plot the training and validation accuracy:
    phase = "Class_naming"
    acc = [0.] + history.history['accuracy']
    val_acc = [0.] + history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plot_performance(phase, acc, val_acc, loss, val_loss)

    # # SHOW PREDICTIONS AFTER TWO EPOCHS FITTING NEW CLASS WEIGHTS
    image_var = tf.Variable(preprocess_input(image_batch))
    pred = model2.predict(image_var)
    pred = np.argmax(pred[0])
    print(pred)

    # # MODEL FINE-TUNING
    model2.trainable = True
    # Fine-tune from this layer onwards
    fine_tune_at = 10
    # Freeze all the layers before the `fine_tune_at` layer
    for layer in model2.layers[0:fine_tune_at]:
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

    fine_tune_epochs = 15
    total_epochs =  initial_epochs + fine_tune_epochs
    print("fine tuning model2 ")
    history_fine = model2.fit(  train_dataset,
                                epochs=total_epochs,
                                initial_epoch=history.epoch[-1],
                                validation_data=validation_dataset,
                                verbose=1)
    print(f"{banner}fine tuning model2 complete in {time.time()-start} seconds")
    model2.save(f'./../MODELS/fungus{datetime.now()}.keras')
    
    # # plot accuracy
    phase = "fine_tuning"
    acc += history_fine.history['accuracy']
    val_acc += history_fine.history['val_accuracy']
    loss += history_fine.history['loss']
    val_loss += history_fine.history['val_loss']
    plot_performance(phase, acc, val_acc, loss, val_loss)


    image_var = tf.Variable(preprocess_input(image_batch))
    pred = model2.predict(image_var)
    pred = np.argmax(pred[0])
    print(pred)
