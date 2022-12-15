from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

iris = load_iris()
x,y = iris.data,iris.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
dtc = tree.DecisionTreeClassifier().fit(x_train, y_train)
pred = dtc.predict(x_test)

plt.figure(figsize=(10,10))
tree.plot_tree(dtc,filled=True)
plt.show()

print(f"Accuracy: {round(accuracy_score(pred, y_test)*100, 2)}%")
