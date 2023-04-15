from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
from sklearn import metrics

# # # Question - 1
# Ignoring the warnings
warnings.filterwarnings("ignore")
# Reading dataset CC.csv
dataset =pd.read_csv(r"CC GENERAL.csv")
# Replacing null values with mean
dataset['MINIMUM_PAYMENTS'].fillna(dataset['MINIMUM_PAYMENTS'].mean(), inplace=True)
dataset['CREDIT_LIMIT'].fillna(dataset['CREDIT_LIMIT'].mean(), inplace=True)
# print(dataset.isnull().any())
x = dataset.iloc[:, 1:18]
# print(x)
# print(dataset['TENURE'].value_counts())
# Applying PCA on CC dataset
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, dataset[['TENURE']]], axis=1)
print('CC dataset finalDf:')
print(finalDf)
X = finalDf.drop('TENURE', axis=1).values
y = finalDf['TENURE'].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)
y_train_hat =logisticRegr.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_hat)*100
print('Accuracy for our Training dataset with PCA is: %.4f %%' % train_accuracy)
# Applying k-means on PCA and Silhoutte score
nclusters = 2
km = KMeans(n_clusters=nclusters)
km.fit(x)
y_cluster_kmeans = km.predict(x)
score = metrics.silhouette_score(x, y_cluster_kmeans)
print('Silhouette Score =', score)
# Performing Scaling+PCA+K-Means and reporting the performance.
# Applying scaling
scaler = StandardScaler()
X_Scale = scaler.fit_transform(x)
# PCA
pca2 = PCA(n_components=2)
principalComponents = pca2.fit_transform(X_Scale)
principalDf_scale = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
finalDf_scale = pd.concat([principalDf_scale, dataset[['TENURE']]], axis=1)
print('CC dataset after scaling finalDf:')
print(finalDf_scale)
nclusters = 2
km = KMeans(n_clusters=nclusters)
km.fit(X_Scale)
y_cluster_kmeans = km.predict(X_Scale)
score = metrics.silhouette_score(X_Scale, y_cluster_kmeans)
print('Silhouette Score after scaling =', score)
"""scaler = StandardScaler()
scaler.fit(x)
x_scaler = scaler.transform(x)
pca = PCA(2)
x_pca = pca.fit_transform(x_scaler)
df2 = pd.DataFrame(data=x_pca)
finaldf = pd.concat([df2, dataset[['TENURE']]], axis=1)
print(finaldf)
nclusters = 2
km = KMeans(n_clusters=nclusters)
km.fit(x_scaler)
y_cluster_kmeans = km.predict(x_scaler)
score = metrics.silhouette_score(x_scaler, y_cluster_kmeans)
print('Silhouette Score after scaling =', score)"""