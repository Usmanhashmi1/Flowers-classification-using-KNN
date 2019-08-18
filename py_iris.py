from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd

#iris = load_iris()
#print iris.feature_names
#print iris.target_names

df = pd.read_csv('final_iris_x.csv')
x = np.asarray(df.iloc[:, :])
df = pd.read_csv('final_iris_y.csv')
y = np.asarray(df.iloc[:, :])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.22, random_state=42)

#print(x_train.shape)
#print(x_test.shape)
#print(y_train.shape)
#print(y_test.shape)



knn = KNeighborsClassifier(n_neighbors=2)

knn.fit(x_train, y_train.ravel())

predicted_y = knn.predict(x_test)
print(predicted_y)
print(y_test.ravel())

print(knn.score(x_test, y_test) * 100)

p = knn.predict([[5, 3, 2, 1], [5, 3, 2, 6]])
print (p)

#knn.score(predicted_y, y_test.ravel())
