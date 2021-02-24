
import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn import metrics

digits = load_digits()
data = scale(digits.data)

y = digits.target
k = len(np.unique(y)) # no. of centroids (chosen by looking at the dataset)
samples, features = data.shape

# scoring the classifier (imported from sklearn)
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))


model = KMeans(n_clusters=k, init="random")  
bench_k_means(model, "1", data)                                   

