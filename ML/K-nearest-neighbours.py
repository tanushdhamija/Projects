
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('car.data')
print(data.head())

# preprocessing data (converting non-numeric values into numeric values)
pre = preprocessing.LabelEncoder()
buying = pre.fit_transform(list(data["buying"]))
maint = pre.fit_transform(list(data["maint"]))
door = pre.fit_transform(list(data["door"]))
persons = pre.fit_transform(list(data["persons"]))
lug_boot = pre.fit_transform(list(data["lug_boot"]))
safety = pre.fit_transform(list(data["safety"]))
clas = pre.fit_transform(list(data["class"]))

predict = "class" #what we want to predict

X = list(zip(buying, maint, door, persons, lug_boot, safety))  # features
y = list(clas)  # labels

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

model = KNeighborsClassifier(n_neighbors=9)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print("\nAccuracy of the model : " ,acc*100, "%\n")

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]
for i in range(10):
    print("Predicted: ", names[predicted[i]], "Actual: ", names[y_test[i]])

