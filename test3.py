#!/usr/bin/env python
# encoding:utf-8

import numpy as np
from sklearn import random_projection
from sklearn.datasets import fetch_mldata
from scipy.spatial import distance

X1 = np.array([[0,1,2],
               [1,0,2],
               [2,3,1]])

X2 = np.array([[0,1,2,2,1],
               [1,0,2,0,1],
               [2,3,1,3,0]])

gram_mat1 = distance.pdist(X1)
gram_mat2 = distance.pdist(X2)
print(gram_mat1)
print(gram_mat2)
gram_mat1 = distance.squareform(gram_mat1)
gram_mat2 = distance.squareform(gram_mat2)
print(gram_mat1)

