# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 00:47:09 2020

@author: Shrikant Agrawal
"""

import matplotlib.pyplot as plt   # For plotting diagrams
import numpy as np                # For creating arrays
import pandas as pd               # To read the dataset
%matplotlib inline


# Import inbuilt dataset
from sklearn.datasets import load_breast_cancer

cancer=load_breast_cancer()
cancer.keys()

print(cancer['DESCR'])  # Output shows number of Attributes are 30 ie columns

#Creating Dataframe
df=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df.head(5)

""" cancer['data'] fetch data and columns gave column names to our dataset
We have 30 different columns and we can convert these 30 columns into 2 using PCA
Before doing PCA standard scalling is mandatory. It is required to rescale the values
of different varialbes into the same unit"""

# from sklearn.preprocessing import MinMaxScaler   OR
from sklearn.preprocessing import StandardScaler # It scale down the value by usding sd =1 and mean=0

scaler=StandardScaler()
scaler.fit(df)
scaled_data=scaler.transform(df)

# Data converted into Array
scaled_data

# Now apply PCA technique to reduce feature to 2
from sklearn.decomposition import PCA

pca=PCA(n_components=2)

pca.fit(scaled_data)

x_pca=pca.transform(scaled_data)

scaled_data.shape

x_pca.shape

scaled_data
x_pca

plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'])
plt.xlabel('First principle component')
plt.ylabel('Second principle component')

# You can apply any Algorithm now, on your target variable