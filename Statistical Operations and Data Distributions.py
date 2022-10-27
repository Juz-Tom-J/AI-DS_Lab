import math as m

dataset = [16,18,21,27,29,30,33,34,39,41,44,44,45,48,54,55,57,59,61,63,66,70,72]
l = len(dataset)

#Mean
mean = sum(dataset)/l
print("Mean:",mean)

#Median
sorted_data = sorted(dataset)
if l%2 == 0: 
    loc1 = int(l/2)
    loc2 = int((l/2)+1)
    median = (dataset[loc1]+dataset[loc2])/2
    print("Median: ",median)
else: 
    loc = int((l+1)/2)
    print("Median: ",sorted_data[loc])

#Mode
d = {}

for i in dataset:
    d[i] = dataset.count(i) 

key_list = list(d.keys())
value_list = list(d.values())

pos = value_list.index(max(value_list))
print("Mode: ",key_list[pos]) 

----------------------------------------------------------------------------------------------------------------------------------------------------------

import statistics as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm,uniform

dataset = [16,18,21,27,29,30,33,34,39,41,44,44,45,48,54,55,57,59,61,63,66,70,72]

print("Mean: ",st.mean(dataset))
print("Median: ",st.median(dataset))
print("Mode: ",st.mode(dataset))
print("Standard Deviation: ",np.std(dataset))
print("Percentile: ",np.percentile(dataset,25))

#Normal distribution
data_norm = np.arange(1,10,0.2)
pdf_n = norm.pdf(data_norm,loc=5,scale=1)
plt.plot(data_norm,pdf_n)
plt.title("Normal Distribution")
plt.xlabel("Data points")
plt.ylabel("Probability Density")
plt.show()

#Uniform distribution
data_uni = np.arange(1,10,0.2)
pdf_u = uniform.pdf(data_uni,loc=5,scale=1)
plt.plot(data_uni,pdf_u)
plt.title("Uniform Distribution")
plt.xlabel("Data points")
plt.ylabel("Probability Density")
plt.show()
