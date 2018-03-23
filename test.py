#!/usr/bin/env python
# encoding:utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from scipy.spatial import distance

def make_gram_matrix(X):
    return distance.pdist(X)

mnist = fetch_mldata("MNIST original", data_home=".")
X = mnist.data[:3000]
gram_mat_X = make_gram_matrix(X)

for n_components in [10, 20, 30, 50, 80, 100, 150, 300, 500]:
    plt.clf()
    rnd = np.random.mtrand._rand
    n_samples = X.shape[0]
    n_features = X.shape[1]
    components = rnd.normal(loc=0.0, scale=1.0 / np.sqrt(n_components), size=(n_components, n_features))
    
    X_new = np.dot(X, components.T)
    gram_mat_X_new = make_gram_matrix(X_new)
    print(gram_mat_X - gram_mat_X_new)
    plt.hist(gram_mat_X - gram_mat_X_new, bins=100)
    plt.title("Random Projection n_components:{}".format(n_components))
    plt.xlabel("Error")
    plt.ylabel("n_samples")
    plt.xlim([-1000, 1000])
    plt.tight_layout()
    plt.legend()
    plt.savefig("images/{}.png".format(n_components))
