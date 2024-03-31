import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

data, meta = arff.loadarff('F:/Weka/Weka-3-8-6/data/weather.nominal.arff')
data_frame = pd.DataFrame(data)

for i in ['outlook', 'temperature', 'humidity', 'windy', 'play']:
    labels = data_frame[i].unique().tolist()
    data_frame[i] = data_frame[i].apply(lambda x: labels.index(x))

data_frame1 = data_frame.drop(columns=['play'])
X_train, X_test, y_train, y_test = train_test_split(data_frame1, data_frame['play'], test_size=0.25, random_state=3)

test = []
for i in range(10):
    tre = DecisionTreeClassifier(criterion='entropy', random_state=1, splitter='random', max_depth=i + 1)
    tre = tre.fit(X_train, y_train)
    score = tre.score(X_test, y_test)
    test.append(score)
plt.plot(range(1, 11), test, color="red", label="max_depth")
plt.legend()
plt.show()

# tre=DecisionTreeClassifier(criterion='entropy',random_state=1,splitter='random',max_depth=3)
tre = DecisionTreeClassifier(criterion='entropy', random_state=1)
tre = tre.fit(X_train, y_train)
print(tre.score(X_test, y_test))

import graphviz
from sklearn.tree import export_graphviz

dot_data = export_graphviz(tre,
                           feature_names=['outlook', 'temperature', 'humidity', 'windy'],
                           filled=True,
                           rounded=True
                           )

graph = graphviz.Source(dot_data)

graph.view()
