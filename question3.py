import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
import seaborn as sns
sns.set(style="white", color_codes=True)

# # # Question -3
# Ignoring the warnings
warnings.filterwarnings("ignore")
# Reading dataset Iris.csv
df = pd.read_csv("iris.csv")
stdsc = StandardScaler()
# Applying LDA
X_train_std = stdsc.fit_transform(df.iloc[:, range(0, 4)].values)
class_le = LabelEncoder()
y = class_le.fit_transform(df['Species'].values)
lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y)
data = pd.DataFrame(X_train_lda)
data['class'] = y
data.columns = ["LD1", "LD2", "class"]
print('Iris dataset after applying LDA:')
print(data)
markers = ['s', 'x', 'o']
colors = ['r', 'b', 'g']
sns.lmplot(x="LD1", y="LD2", data=data, hue='class', markers=markers, fit_reg=False, legend=False)
plt.legend(loc='upper center')
plt.show()

# # # Question - 4
# PCA performs better in case where number of samples per class is less,
# whereas LDA works better with large dataset having multiple classes.
# PCA is an unsupervised learning algorithm while LDA is a supervised learning algorithm.