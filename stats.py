import pandas as pd
data = pd.read_csv("data.csv")
print('Iris-setosa')
setosa = data['species'] == 'Iris-setosa'
print(data[setosa].describe())
print('\nIris-versicolor')
setosa = data['species'] == 'Iris-versicolor'
print(data[setosa].describe())
print('\nIris-virginica')
setosa = data['species'] == 'Iris-virginica'
print(data[setosa].describe())