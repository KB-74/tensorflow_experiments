# TensorFlow and tf.keras
# Helper libraries
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

# importeer de mnist bibliotheek met testafbeeldingen
fashion_mnist = keras.datasets.fashion_mnist

# Verdeel de dataset in een trainings set en een test set.
# de trainings set heeft 60000 records en de test set heeft er 10000
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# delen door 255 zodat de greysclae pixelwaredes nu tussen de 0 en 1 zitten
train_images = train_images / 255.0
test_images = test_images / 255.0

print('shape of training set is: ' + str(train_images.shape))
print('shape of one image of the training set is: ' + str(train_images[0].shape))
print('shape of test set is: ' + str(test_images.shape))

# bijbehoorende catahorie namen
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Computational machine learning model met twee hidden layers van 128 en 10 neurons
model = keras.Sequential([
    # De eerste laag in het model verandert de 2d input naar een 1d array
    keras.layers.Flatten(input_shape=(28, 28)),
    # eerste hidden layer met 128 neurons
    keras.layers.Dense(128, activation=tf.nn.relu),
    # tweede hidden layer met 10 neurons
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

model.save_weights('./models/kleren.h5')

predictions = model.predict(test_images)


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

jpgfile = Image.open("./images/test_4.jpg").convert('L')
image = np.array(jpgfile.getdata()).reshape(1, jpgfile.size[0], jpgfile.size[1])
image = image / -255.0

i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)

num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
# for i in range(num_images):
#     plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
#     plot_image(i, predictions, test_labels, test_images)
#     plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
#     plot_value_array(i, predictions, test_labels)

plt.show()
plt.close()
