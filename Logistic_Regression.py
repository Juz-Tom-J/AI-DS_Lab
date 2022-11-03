import numpy as np
from sklearn import linear_model

X = np.array([3.78,2.44,2.09,0.14,1.72,1.65,4.92,4.37,4.96,4.52,3.69,5.88]).reshape(-1,1)
y = np.array([0,0,0,0,0,0,1,1,1,1,1,1])

logr = linear_model.LogisticRegression()
logr.fit(X,y)

tumor = input("Enter the tumor size: ")
predicted = logr.predict(np.array([tumor]).reshape(-1,1))
if predicted == 1:
    print('Cancerous') 
else:
    print('Non-Cancerous')
