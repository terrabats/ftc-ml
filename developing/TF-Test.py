import tensorflow as tf
import numpy as np
from tensorflow import keras



x = [-1, 0, 1, 2, 3, 4]
y = [-3, -1, 1, 3, 5, 7]

model = keras.models.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(x, y, epochs=500)

print(model.predict([10]))


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()


with tf.io.gfile.GFile('test.tflite', 'wb') as f:
  f.write(tflite_model)