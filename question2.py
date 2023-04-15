from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import warnings

# # # Question - 2
# Ignoring the warnings
warnings.filterwarnings("ignore")
# Reading dataset pd_speech_features.csv
df = pd.read_csv("pd_speech_features.csv")
X = df.drop('class', axis=1).values
y = df['class'].values
# Apply scaling
scaler = StandardScaler()
X_Scale = scaler.fit_transform(X)
# PCA
pca2 = PCA(n_components=3)
principalComponents = pca2.fit_transform(X_Scale)
principalDf = pd.DataFrame(data=principalComponents,
                           columns=['principal component 1', 'principal component 2', 'principal component 3'])
finalDf = pd.concat([principalDf, df[['class']]], axis=1)
print('PD_Speech_Features dataset finalDf:')
print(finalDf)
# SVM
X_train, X_test, y_train, y_test = train_test_split(X_Scale, y, test_size=0.3, random_state=0)
svc = SVC(max_iter=1000)
svc.fit(X_train, y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, y_train) * 100, 2)
print("SVM accuracy =", acc_svc)