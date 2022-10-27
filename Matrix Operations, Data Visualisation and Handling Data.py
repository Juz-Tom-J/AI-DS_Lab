import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

m1 = np.arange(1,10).reshape(3,3)
m2 = np.arange(11,20).reshape(3,3)

print("Matrix 1: \n",m1)
print("Matrix 2: \n",m2)

print("Sum: \n",np.add(m1,m2))
print("Difference: \n",np.subtract(m2,m1))
print("Product: \n",np.dot(m1,m2))
print("On dividing: \n",np.divide(m2,m1))
print("Transpose of matrix 1: \n",np.transpose(m1))

#Dataset
x = np.arange(0,10)
y = np.arange(11,21)


#Plotting using matplotlib
plt.scatter(x,y,c='b')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Plotting using MatPlotLib')
plt.show()

#Plotting using seaborn
graph = sb.stripplot(x)
graph.set(xlabel='X axis',ylabel='Y axis',title='Plotting using Seaborn')
plt.show()

#Using pandas
df = pd.read_csv('advertising.csv')
df.info()
