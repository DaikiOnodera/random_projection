#!/usr/bin/env python
# encoding:utf-8

import numpy as np
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata("MNIST original", data_home=".")
X = mnist.data
y = mnist.target

rnd = np.random.mtrand._rand
n_samples = X.shape[0]
n_components = 10
n_features = X.shape[1]
print(rnd.normal(loc=0.0, scale=1.0 / np.sqrt(n_components), size=(n_components, n_features)).shape)
