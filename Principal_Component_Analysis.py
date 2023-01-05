from sklearn import datasets
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
import seaborn as sns 
import matplotlib.pyplot as plt

iris = datasets.load_iris()
df = pd.DataFrame(iris['data'], columns = iris['feature_names'])

scalar = StandardScaler().fit_transform(df)
scaled_data = pd.DataFrame(scalar)

pca = PCA(n_components = 3).fit(scaled_data) 
data_pca = pca.transform(scaled_data)
data_pca = pd.DataFrame(data_pca,columns=['PC1','PC2','PC3'])

print("Correlation before PCA:\n",scaled_data.corr())
sns.heatmap(scaled_data.corr())
plt.show()

print("Correlation after PCA:\n",data_pca.corr())
sns.heatmap(data_pca.corr()) 
plt.show()
