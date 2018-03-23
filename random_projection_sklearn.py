#!/usr/bin/env python
# encoding:utf-8

import numpy as np
from sklearn import random_projection
from sklearn.datasets import fetch_mldata
from scipy.spatial import distance

mnist = fetch_mldata("MNIST original", data_home=".")
X = mnist.data[:3000]
y = mnist.target[:3000]

gram_mat = distance.squareform(distance.pdist(X))
print(gram_mat)

print("original shape:{}".format(X.shape))
transformer = random_projection.GaussianRandomProjection(n_components=10)
X_new = transformer.fit_transform(X)
print("transformed shape:{}".format(X_new.shape))

