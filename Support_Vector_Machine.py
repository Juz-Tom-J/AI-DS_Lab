import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score

heart = pd.read_csv('SVM_heart_dis.csv')

labels= np.array(heart.loc[:, 'target'])
features= np.array(heart.iloc[:, :13])

x_train,x_test,y_train,y_test = train_test_split(features[:250],labels[:250],test_size=0.2,random_state=62)
svm_linear = SVC(kernel='linear', C=0.01).fit(x_train, y_train)
pred = svm_linear.predict(x_test)

cm = confusion_matrix(y_test, pred)
cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["Disease_Positive","Disease_Negative"])
cm_disp.plot()
plt.show()
print("Accuracy:", accuracy_score(y_test,pred)*100)
