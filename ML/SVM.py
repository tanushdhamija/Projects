
import sklearn
from sklearn import svm
from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


cancer = datasets.load_breast_cancer() #cancer dataset from sklearn

#print("Features: ", cancer.feature_names) # Features of the dataset
#print("\nLabels: ", cancer.target_names) # Labels of the dataset

X = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

classes = ['malignant' , 'benign']

model_SVM = svm.SVC(kernel="linear")
model_KNN = KNeighborsClassifier(n_neighbors=9)
model_SVM.fit(x_train, y_train)
model_KNN.fit(x_train, y_train)


y_pred_SVM = model_SVM.predict(x_test)
y_pred_KNN = model_KNN.predict(x_test)

acc_SVM = metrics.accuracy_score(y_test, y_pred_SVM)
acc_KNN = metrics.accuracy_score(y_test, y_pred_KNN)


print("\nAccuracy of SVM : " ,acc_SVM*100, "%")
print("\nAccuracy of KNN : " ,acc_KNN*100, "%")


