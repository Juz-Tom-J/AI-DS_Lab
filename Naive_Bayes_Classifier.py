from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
b_cancer = datasets.load_breast_cancer()

X = b_cancer.data
y = b_cancer.target
  
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
import seaborn as sn
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n', cm)
sn.heatmap(cm,annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

TN =  cm[0,0]
TP =  cm[1,1]
FP =  cm[0,1]
FN =  cm[1,0]

accuracy = float((TP+TN)/(TP+FN+TN+FP))
print("Accuracy = ",accuracy)
precision = float(TP/(TP+FP))
print("Precision = ",precision)
recall = float(TP/(TP+FN))
print("Recall = ",recall)
