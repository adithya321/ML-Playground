import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

test_idx = [0, 50, 100]

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# test data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

print test_target
print clf.predict(test_data)

# viz code
from sklearn.externals.six import StringIO
import pydotplus

dot_data = StringIO()
tree.export_graphviz(clf,
                     out_file=dot_data,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True, rounded=True,
                     impurity=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")

print test_data[2], test_target[2]

print iris.feature_names
print iris.target_names

x = iris.data
y = iris.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)

decisionTreeClassifier = tree.DecisionTreeClassifier()
decisionTreeClassifier.fit(x_train, y_train)
decisionTreeClassifierPredictions = decisionTreeClassifier.predict(x_test)
print decisionTreeClassifierPredictions

from sklearn.neighbors import KNeighborsClassifier

kNeighboursClassifier = KNeighborsClassifier()
kNeighboursClassifier.fit(x_train, y_train)
kNeighboursClassifierPredictions = kNeighboursClassifier.predict(x_test)
print kNeighboursClassifierPredictions

from sklearn.metrics import accuracy_score

print "accuracy_score"
print "decisionTreeClassifierPredictions"
print accuracy_score(y_test, decisionTreeClassifierPredictions)
print "kNeighboursClassifierPredictions"
print accuracy_score(y_test, kNeighboursClassifierPredictions)
