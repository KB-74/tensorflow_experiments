# import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


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

model.load_weights('./models/kleren.h5')

jpgfile = Image.open("./images/test_4.jpg").convert('L')
image = np.array(jpgfile.getdata()).reshape(1, jpgfile.size[0], jpgfile.size[1])
image = image / -255.0

plt.figure()
plt.imshow(image[0])
plt.colorbar()
plt.grid(False)
plt.show()
plt.close()

prediction = model.predict(image)

print(class_names[list(prediction[0]).index(max(prediction[0]))])
