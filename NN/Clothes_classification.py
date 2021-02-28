import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

def visualize_input(train_images, train_labels):
    plt.figure(figsize=(8,8))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()
#visualize_input(train_images, train_labels)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_images, train_labels, epochs=5)

predictions = model.predict(test_images)

plt.figure(figsize=(5,5))
r = np.random.randint(0,100)
for i in range(5):
    plt.xticks([])
    plt.yticks([])
    plt.imshow(test_images[i*r], cmap=plt.cm.binary)
    plt.xlabel("Actual : " + class_names[test_labels[i*r]])
    plt.title("Prediction : " + class_names[np.argmax(predictions[i*r])])
    plt.show()
