# TensorFlow and tf.keras
# Helper libraries
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# importeer de mnist bibliotheek met testafbeeldingen
fashion_mnist = keras.datasets.fashion_mnist

# Verdeel de dataset in een trainings set en een test set.
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print('shape of training set is: ' + str(train_images.shape))
print('shape of test set is: ' + str(train_images.shape))
'''
train_images = train_images / 255.0
test_images = test_images / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print('The shape of the data set is ' + str(train_images.shape))

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

# maken van voorspellingen
predictions = model.predict(test_images)
'''