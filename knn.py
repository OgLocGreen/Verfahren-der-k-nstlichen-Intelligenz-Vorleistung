# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import gc
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
#from tensorflow.keras.utils import plot_model
import IPython.display as display
from PIL import Image
import matplotlib.pyplot as plt
#from keras.utils.vis_utils import model_to_dot

import numpy as np
import pandas as pd
import time
import pathlib

# GPU auf maximal 30% Leistung
"""
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = False
session = tf.Session(config=config)
keras.backend.set_session(session)
"""

import os

def train_and_evalu(gene,knn = "fully", gpu=False, f1=False):

    if gpu:
        pass
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    ### Hyperparameter 
    var_learningrate,var_dropout,var_epoch,var_batch_size,optimizer = gene[0],gene[1],gene[2],gene[3],gene[4]

    print("var_learningrate", var_learningrate, "var_dropout", var_dropout, "var_epoch", var_epoch, "var_batch_size", var_batch_size, "optimizer", optimizer)

   
      
    ### hier müssen die Daten Spoon Fork Kniev ausgewählt werden

    data_dir = "./dataset_fsk/"
    data_dir = pathlib.Path(data_dir)
    image_count = len(list(data_dir.glob('*/*.jpg')))
    CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
    print("CLASS_NAMES",CLASS_NAMES)

    fork = list(data_dir.glob('fork/*'))

    for image_path in fork[:3]:
        display.display(Image.open(str(image_path)))

    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    BATCH_SIZE = 32
    IMG_HEIGHT = 200
    IMG_WIDTH = 200
    STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

    train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes = list(CLASS_NAMES))


    image_batch, label_batch = next(train_data_gen)
    show_batch(image_batch, label_batch,CLASS_NAMES)
    

    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))


    #train_images = train_images / 255.0
    #test_images = test_images / 255.0


    ### Selection of the Model
    try:
        tf.compat.v1.set_random_seed(1)
    except:
        tf.random.set_seed(1)
    if knn == "fully":
        model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=train_images.shape[1:]),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(var_dropout),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(var_dropout),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(var_dropout),
        keras.layers.Dense(10, activation='softmax')
            ])
    elif knn == "cnn":
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(32, (3, 3), padding='same',
                        input_shape=train_images.shape[1:]))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Conv2D(32, (3, 3)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(var_dropout))

        model.add(keras.layers.Conv2D(64, (3, 3), padding='same'))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Conv2D(64, (3, 3)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(var_dropout))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dropout(var_dropout))
        model.add(keras.layers.Dense(10))
        model.add(keras.layers.Activation('softmax')) 

    ### Optimizer
    adam = keras.optimizers.Adam(lr=var_learningrate)
    Adagrad = keras.optimizers.Adagrad(lr=var_learningrate)
    RMSprop = keras.optimizers.RMSprop(lr=var_learningrate)
    SGD = keras.optimizers.SGD(lr=var_learningrate)
    adadelta = keras.optimizers.Adadelta(learning_rate=var_learningrate)
    adammax = keras.optimizers.Adamax(learning_rate=var_learningrate)
    nadam = keras.optimizers.Nadam(learning_rate=var_learningrate)
    ftrl = keras.optimizers.Ftrl(learning_rate=var_learningrate)
    optimizerarray = [adam, SGD, RMSprop, Adagrad, adadelta,adammax,nadam,ftrl]

    if round(optimizer) < -0.5:
        optimizer = 0
    elif round(optimizer) > 7.5:
        optimizer = 7
        
    if knn == "fully":
        model.compile(optimizer=optimizerarray[round(optimizer)],
                    loss='sparse_categorical_crossentropy',
                    metrics=['acc'])
    elif knn == "cnn":
        model.compile(loss='categorical_crossentropy',
              optimizer=optimizerarray[round(optimizer)],
              metrics=['acc'])

    ### Model fit and Evaluate
    mittel = 3
    loss = 0
    acc = 0
    test_loss = 0
    test_acc = 0 
    model.fit(train_images, train_labels, epochs=int(var_epoch),batch_size=int(var_batch_size),use_multiprocessing=True, workers=2,verbose=0)
    for x in range(mittel):
        test_loss,test_acc = model.evaluate(eval_images, eval_labels,verbose=0)
        loss += test_loss
        acc += test_acc
    test_acc = acc/mittel
    test_loss = loss/mittel
    #plot_model(model, to_file='model_cnn.png')
    variables = 0
    variables = np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])

#%%
    if f1:
        y_pred = model.predict(test_images)
        y_pred_bool = np.argmax(y_pred, axis=1)
        if cnn:
            print("cnn",cnn)
            test_labels_bool = np.argmax(test_labels, axis=1)
            precision_score_var = precision_score(test_labels_bool, y_pred_bool , average="macro")
            recall_score_var = recall_score(test_labels_bool, y_pred_bool , average="macro")
            f1_score_var = f1_score(test_labels_bool, y_pred_bool , average="macro")
            cm  = confusion_matrix(y_true = test_labels_bool, y_pred = y_pred_bool)
        elif fully:
            print("fully",fully)
            precision_score_var = precision_score(test_labels, y_pred_bool , average="macro")
            recall_score_var = recall_score(test_labels, y_pred_bool , average="macro")
            f1_score_var = f1_score(test_labels, y_pred_bool , average="macro")
            cm  = confusion_matrix(y_true = test_labels, y_pred = y_pred_bool)

            print("test_loss: ",test_loss , "test_acc: ", test_acc, "variables",variables, "precision_score_var", precision_score_var,"recall_score_var", recall_score_var,"f1_score_var", f1_score_var)
            
        return test_loss, test_acc, variables, precision_score_var, recall_score_var, f1_score_var, cm

    print("test_loss: ",test_loss , "test_acc: ", test_acc, "variables",variables)
    gc.collect()
    return test_loss, test_acc, variables

def show_batch(image_batch, label_batch,CLASS_NAMES):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
      plt.axis('off')

def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  return parts[-2] == CLASS_NAMES

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

#%%
if __name__ == "__main__":
    #%%
    #gene = [var_learningrate = 0.05,var_dropout=0.25,var_epoch=100,var_batch_size=16,optimizer=3]
    gene= [0.05,0.25,10,16,3]
    start = time.time()
    test_loss, test_acc, variables, precision_score_var, recall_score_var, f1_score_var, cm = train_and_evalu(gene,f1 = True)
    end = time.time() - start
    print(end)

    start = time.time()
    gene= [0.05,0.25,80,16,3]
    start = time.time()
    test_loss, test_acc, variables, precision_score_var, recall_score_var, f1_score_var, cm = train_and_evalu(gene,f1 = True)
    end = time.time() - start
    print(end)

    #%%
    df_cm = pd.DataFrame(cm)
    plt.figure()
    sn.heatmap(df_cm, annot=True, fmt="d")
    plt.show()