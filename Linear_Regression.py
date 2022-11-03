import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd

df=pd.read_csv('Iris.csv')

X= df.loc[:, 'SepalLengthCm']
Y= df.loc[:, 'SepalWidthCm']

m,c,r,p,std_err = st.linregress(X,Y)

def y_eq(X):
    return (m*X)+c

model = list(map(y_eq,X))
plt.title("Iris Data")
plt.scatter(X,Y)
plt.plot(X,model)
plt.xlabel("Sepal Length in cm")
plt.ylabel("Sepal Width in cm")
plt.show()

predict = float(input("Enter Value of sepal length to predict the sepal width\t"))
print("Value predicted: ",y_eq(predict))
