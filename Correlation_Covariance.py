from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd

b_cancer = load_breast_cancer()
df = pd.DataFrame(data=b_cancer.data, columns=b_cancer.feature_names)
df.head()

sam1 = df.loc[:, "mean radius"]
sam2 = df.loc[:, "mean perimeter"]

cov = round(np.cov(sam1, sam2)[1][0], 4)
cor = round(np.corrcoef(sam1, sam2)[1][0], 4)
print(f"Mean Radius and Mean Perimeter of Cancer\nCovariance: {cov}\nCorrelation:{cor}")
