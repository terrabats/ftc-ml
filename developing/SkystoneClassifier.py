import numpy as np
from tensorflow.keras.preprocessing import image
import os
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import time

startT = time.time()


train_images = []
X1 = []
y1 = []
foldername = 'train_ims/n/'
for i in os.listdir(f'{foldername}'):
    img = image.load_img(f'{foldername}{i}')
    img = img.resize((128, 128))
    img = image.img_to_array(img)
    img = img/255
    train_images.append(img)
    y1.append([1.0,0.0])
X1 = np.array(train_images)
y1 = np.array(y1)


train_images = []
X2 = []
y2 = []
foldername = 'train_ims/s/'
for i in os.listdir(f'{foldername}'):
    img = image.load_img(f'{foldername}{i}')
    img = img.resize((128, 128))
    img = image.img_to_array(img)
    img = img/255
    train_images.append(img)
    y2.append([0.0,1.0])
X2 = np.array(train_images)
y2 = np.array(y2)


X = np.concatenate((X1, X2))
y = np.concatenate((y1,y2))

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
X_train = X
y_train = y

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(128,128,3)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='Adam')
model.fit(X_train, y_train, epochs=4)

foldername = 'train_ims/n/'
test_img = image.load_img(f'{foldername}167.png')
test_img = test_img.resize((128, 128))
test_img = image.img_to_array(test_img)
test_img = test_img/255
test_img = np.array([test_img])

pred = np.argmax(model.predict(test_img),axis=-1)
print(pred)


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()


with tf.io.gfile.GFile('tf_models/class.tflite', 'wb') as f:
  f.write(tflite_model)


endT = time.time()
print(f'The neural net has taken {(endT - startT)/60} minutes to train.')



