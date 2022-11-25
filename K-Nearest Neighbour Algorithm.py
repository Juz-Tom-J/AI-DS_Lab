from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
import numpy as np
import matplotlib.pyplot as plt

wineData = load_wine()

X = wineData.data
y = wineData.target

X_train,X_test,y_train,y_test = train_test_split(
X,y,test_size = 0.2,random_state=42)

k = int(input("Enter the value for k: "))
knn = KNeighborsClassifier(n_neighbors=k)
print("Number of data records for prediction is: ",len(X_test))

knn.fit(X_train,y_train)
print("Prediction Accuracy:",knn.score(X_test, y_test)*100)

neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i,j in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=j)
    knn.fit(X_train,y_train)
    train_accuracy[i] = knn.score(X_train,y_train)
    test_accuracy[i] = knn.score(X_test,y_test)

plt.plot(neighbors, test_accuracy, label = 'Testing Dataset Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Dataset Accuracy')
  
plt.legend()
plt.xlabel('n neighbors')
plt.ylabel('Accuracy')
plt.show()
