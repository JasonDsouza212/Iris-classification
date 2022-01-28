import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.linear_model import SGDClassifier

iris = datasets.load_iris()

X = iris.data
y = iris.target_names[iris.target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

sgd = SGDClassifier(random_state=42)
sgd.fit(X_train, y_train)
y_pred = sgd.predict(X_train)

conf_matrix = confusion_matrix(y_true=y_train, y_pred=y_pred)
conf_matrix