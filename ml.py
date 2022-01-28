import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas.plotting import parallel_coordinates
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('data.csv')

data.head(5)

data.describe()

data.groupby('species').size()

train, test = train_test_split(data, test_size = 0.4, stratify = data["species"], random_state = 42)
print(train)

n_bins = 10
fig, axs = plt.subplots(2, 2)
axs[0,0].hist(train['sepal_length'], bins = n_bins)
axs[0,0].set_title('Sepal Length')
axs[0,1].hist(train['sepal_width'], bins = n_bins)
axs[0,1].set_title('Sepal Width')
axs[1,0].hist(train['petal_length'], bins = n_bins)
axs[1,0].set_title('Petal Length')
axs[1,1].hist(train['petal_width'], bins = n_bins)
axs[1,1].set_title('Petal Width')
# add some spacing between subplots
fig.tight_layout(pad=1.0)
plt.title("label")
plt.show()


fig, axs = plt.subplots(2, 2)
fn = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
cn = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
sns.boxplot(x = 'species', y = 'sepal_length', data = train, order = cn, ax = axs[0,0])
sns.boxplot(x = 'species', y = 'sepal_width', data = train, order = cn, ax = axs[0,1])
sns.boxplot(x = 'species', y = 'petal_length', data = train, order = cn, ax = axs[1,0])
sns.boxplot(x = 'species', y = 'petal_width', data = train,  order = cn, ax = axs[1,1])


# add some spacing between subplots
fig.tight_layout(pad=1.0)
plt.show()

sns.violinplot(x="species", y="petal_length", data=train, size=5, order = cn, palette = 'colorblind')
plt.show()

sns.pairplot(train, hue="species", height = 2, palette = 'colorblind')
plt.show()

corrmat = train.corr()
sns.heatmap(corrmat, annot = True, square = True)
plt.show()

parallel_coordinates(train, "species", color = ['blue', 'red', 'green'])
plt.show()

X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species
plt.show()

mod_dt = DecisionTreeClassifier(max_depth = 3, random_state = 1)
mod_dt.fit(X_train,y_train)
prediction=mod_dt.predict(X_test)
plt.show()
mod_dt.feature_importances_


plt.figure(figsize = (10,8))
plot_tree(mod_dt, feature_names = fn, class_names = cn, filled = True)
plt.show()

disp = metrics.plot_confusion_matrix(mod_dt, X_test, y_test, display_labels=cn, cmap=plt.cm.Blues, normalize=None)
disp.ax_.set_title('Decision Tree Confusion matrix, without normalization')
plt.show()