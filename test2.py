#!/usr/bin/env python
# encoding:utf-8

import numpy as np
from sklearn import random_projection
from sklearn.datasets import fetch_mldata
from scipy.spatial import distance

X = np.array([[0,1,2],
              [1,0,2],
              [2,3,1]])

gram_mat = distance.squareform(distance.pdist(X))
print(gram_mat)

