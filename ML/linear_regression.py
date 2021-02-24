
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
#from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

X = np.array(data.drop([predict], 1)) # Features
y = np.array(data[predict]) # Labels

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test) # acc stands for accuracy
print("Accuracy of the model : " , acc*100,"%")

predictions = linear.predict(x_test)

for i in range(5):
    print("\nPredicted grade : " , predictions[i], "Actual grade : " ,y_test[i])