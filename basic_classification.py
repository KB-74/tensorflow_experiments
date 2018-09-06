# TensorFlow and tf.keras
# Helper libraries
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# importeer de mnist bibliotheek met testafbeeldingen
fashion_mnist = keras.datasets.fashion_mnist

# Verdeel de dataset in een trainings set en een test set.
# de trainings set heeft 60000 records en de test set heeft er 10000
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# delen door 255 zodat de greysclae pixelwaredes nu tussen de 0 en 1 zitten
train_images = train_images / 255.0
test_images = test_images / 255.0

print('shape of training set is: ' + str(train_images.shape))
print('shape of test set is: ' + str(test_images.shape))

# bijbehoorende catahorie namen
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Computational machine learning model met twee hidden layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# model compileren
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# het trainene van het machine learning model
model.fit(train_images, train_labels, epochs=5)

# getting all the metrics
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)
