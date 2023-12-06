#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


import os
for dirname, _,filenames in os.walk('/home/scis/Downloads/mri image YN'):
    for filename in filenames:
        print(os.path.join(dirname,filename))


# In[4]:


#for the model
import keras
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense,MaxPooling2D,Dropout
from keras import backend as k


# In[5]:


#For laoding dataset,preprocessing and train_test split
import ipywidgets as widgets
import io
from PIL import Image
import tqdm
from sklearn.model_selection import train_test_split
import cv2
from sklearn.utils import shuffle
import tensorflow as tf


# In[6]:


#For data visualation
import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


#Load/read the dataset
labels = ['Tumor','notumor']
image_size = 150
train_folder_path = '/home/scis/Downloads/mri image YN/Traning'
test_folder_path = '/home/scis/Downloads/mri image YN/Testing'


# In[8]:


#load the training dataset
train_images = []
train_labels = []
for i, label in enumerate(labels):
    folder_path = os.path.join(train_folder_path, label)
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        img = cv2.resize(img, (image_size, image_size))
        train_images.append(img)
        train_labels.append(i)


# In[10]:


#train_labels


# In[11]:


len(train_images), len(train_labels)


# In[12]:


#Load the datasert for use in model.evaluate
test_images =[]
test_labels = []
for i, label in enumerate(labels):
    folder_path = os.path.join(test_folder_path, label)
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        img = cv2.resize(img, (image_size, image_size))
        test_images.append(img)
        test_labels.append(i)


# In[14]:


#test_labels


# In[15]:


len(test_images), len(test_labels)


# In[16]:


#converting the image data and label arrays to numpy arrrays
X_train = np.array(train_images)
Y_train = np.array(train_labels)
X_test = np.array(test_images)
Y_test = np.array(test_labels)


# In[17]:


# #converting labels arrays to integers
# Y_train = Y_train.astype(int)
# Y_test = X_test.astype(int)

# #shuffle trainig dataset
# X_train,Y_train = shuffle(X_train,Y_train,random_state=101)
# X_train.shape


# In[18]:


# #preprocessing
# from keras.preprocessing.image import ImageDataGenerator

# #Define folder paths
# train_dir = '/home/scis/Downloads/mri image YN/Traning'
# test_dir = '/home/scis/Downloads/mri image YN/Testing'

# #define image dimension
# img_height = 128
# img_width = 128

# #Define ImageDAtaGenerator for training data
# train_datagen = ImageDataGenerator(
#      # Data augmentation and rescaling parameters
#     rotation_range=20,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     vertical_flip=False,
#     rescale=1./255
# )

# # Define ImageDataGenerator for test data
# test_datagen = ImageDataGenerator(rescale=1./255)

# # Generate data batches from folders; target size affects the size of imgs in data batches...
# #...but does not affect the actual image data that is loaded from the disk...
# #...actual img from disk (that is fed in model) is affected by image size = 150 in preprocessing
# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(img_height, img_width),
#     class_mode='categorical')

# test_generator = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=(img_height, img_width),
#     class_mode='categorical')


# In[19]:


#Y_train


# In[20]:


#train-test split
#X_train,X_test,Y_train,Y_test = train_test_split(X_train,Y_train,test_size=0.1,random_state=101)

#convert integer labels to one-hot encoded categorical labels as requireds by categoriacal cross-entrop y loss
Y_train = tf.keras.utils.to_categorical(Y_train,num_classes = len(labels))
Y_test = tf.keras.utils.to_categorical(Y_test, num_classes = len(labels))


# In[22]:


#Y_train


# In[23]:


len(labels) #len(Y_train)


# In[24]:


Y_train = np.asarray(train_labels).astype('float32').reshape((-1,1))
Y_test = np.asarray(test_labels).astype('float32').reshape((-1,1))


# In[25]:


# Proposed 10-Layer Model
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


# In[26]:


model.summary()


# In[27]:


#Def precision ,Recall and F1 score
from keras.metrics import Precision,Recall
from keras import backend as k


# In[152]:


# def f1_score(y_true,y_pred):
#     y_true=k.round(y_true)
#     y_prep = K.round(y_Pred)
#     tp = k.sum(y_true * y_pred)
#     fd = k.sum((1-y_true) * y_pred)
#     fn = k.sum(y_true * (1-y_pred))
#     precion = tp / (tp + fn + K.epsilon())
#     recall= tp/(tp+fn + K.epsilon)
#     #f1_score = 2*((precion*recall)/(precison+recall+K.epsilon()))
#     return f1_score


# In[28]:


# Early Stopping
from keras.callbacks import EarlyStopping
#early_stopping = EarlyStopping(monitor= 'val_loss', patience = 3)

# Model Compilation
from keras.optimizers import Adam
model.compile(loss= 'binary_crossentropy',optimizer= Adam(learning_rate= 0.001),metrics=['accuracy', Precision(), Recall()])


# In[29]:


#model.save('projectbraintumor.h5')


# In[31]:


len(X_train), len(Y_train)
len(X_test), len(Y_test)


# In[ ]:


# Model Training
history = model.fit(X_train, Y_train, batch_size = 64, epochs= 20, validation_split=0.1)


# In[33]:


len(X_test),len(Y_test)


# In[37]:


# Model Evaluation
scores = model.evaluate(X_test, Y_test)
# print('Test accuracy:', scores[1])
# print('Test F1-score:', scores[2])
# print('Test precision:', scores[3])
# print('Test recall:', scores[4])


# In[38]:


# Graphs of Training and Validation Accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(len(acc)) # def range of vakues for x-axis as number of epochs
fig = plt.figure(figsize=(14,7))
plt.plot(epochs, acc,'r',label="Training Accuracy")
plt.plot(epochs, val_acc,'b',label="Validation Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.savefig('Graphs of Training and Validation Accuracy.png') # to download img
plt.show()


# In[ ]:




