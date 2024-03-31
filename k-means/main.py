import pandas as pd
from scipy.io import arff
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data, meta = arff.loadarff('F:/Weka/Weka-3-8-6/data/iris.arff')
data_frame = pd.DataFrame(data)
print(data_frame)

# Fonctions d'apprentissage automatique qui ne peut gérer que des types numériques
labels = data_frame['class'].unique().tolist()
data_frame['class'] = data_frame['class'].apply(lambda x: labels.index(x))

# Combinez toutes les données 2e et 3e dimensions dans un seul bloc de données
x = data_frame.iloc[:, [2, 3]]
plt.scatter(x['petallength'], x['petalwidth'], c="red", marker='o', label='see')
plt.xlabel('petallength')
plt.ylabel('petalwidth')
plt.legend(loc=2)  # upper left
plt.show()

# Renvoyez l'étiquette correspondant à chaque donnée et mappez la valeur de l'étiquette au cluster correspondant
k_means = KMeans(n_clusters=3)
y_means = k_means.fit(x)  # fit() predict()
label_pred = y_means.labels_

x0 = x[label_pred == 0]
x1 = x[label_pred == 1]
x2 = x[label_pred == 2]
plt.scatter(x0['petallength'], x0['petalwidth'], c="red", marker='o', label='label0')
plt.scatter(x1['petallength'], x1['petalwidth'], c="yellow", marker='+', label='label1')
plt.scatter(x2['petallength'], x2['petalwidth'], c="blue", marker='*', label='label2')
plt.xlabel('petallength')
plt.ylabel('petalwidth')
plt.legend(loc=2)
plt.show()

'''机器学习的函数,只能处理数字类型 weather.nominal'''
'''for i in ['outlook', 'temperature', 'humidity', 'windy', 'play']:
    labels = data_frame[i].unique().tolist()
    data_frame[i] = data_frame[i].apply(lambda x: labels.index(x))

data_frame1 = data_frame.drop(columns=['play'])
k_means = KMeans(n_clusters=5)
y_means = k_means.fit_predict(data_frame1)'''
'''Fit_predict: 返回每个数据对应的标签,并将标签值对应到相应的簇'''
'''print("\n", y_means)'''
