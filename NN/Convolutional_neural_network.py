import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

data = tf.keras.datasets.mnist


(x_train, y_train), (x_test, y_test) = data.load_data()
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

#print(x_train.shape[1:])
#plt.imshow(x_train[5], cmap=plt.cm.binary)
#plt.show()

model = tf.keras.Sequential([

		# first convolutional layer
		tf.keras.layers.Conv2D(32, (3,3), input_shape= (28,28,1), activation='relu'),
		tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

		# second convolutional layer
		tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
		tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

		# fully connected classifier
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(1024, activation='relu'),
		tf.keras.layers.Dense(10, activation='softmax')
		])


model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train)
predictions = model.predict(x_test)

plt.figure(figsize=(5,5))
r = np.random.randint(0,100)
for i in range(5):
    plt.grid(False)
    plt.imshow(x_test[i*r], cmap=plt.cm.binary)
    plt.xlabel("Actual : " + class_names[y_test[i*r]])
    plt.title("Prediction : " + class_names[np.argmax(predictions[i*r])])
    plt.show()




