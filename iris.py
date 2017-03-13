import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from scipy.spatial import distance


def euclidean_distance(a, b):
    return distance.euclidean(a, b)


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

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=.5)

decisionTreeClassifier = tree.DecisionTreeClassifier()
decisionTreeClassifier.fit(X_train, Y_train)
decisionTreeClassifierPredictions = decisionTreeClassifier.predict(X_test)
print decisionTreeClassifierPredictions

from sklearn.neighbors import KNeighborsClassifier

kNeighboursClassifier = KNeighborsClassifier()
kNeighboursClassifier.fit(X_train, Y_train)
kNeighboursClassifierPredictions = kNeighboursClassifier.predict(X_test)
print kNeighboursClassifierPredictions


class ScrappyKNN:
    def __init__(self):
        self.X_train = None
        self.Y_train = None

    def fit(self, x_train, y_train):
        self.X_train = x_train
        self.Y_train = y_train

    def predict(self, x_test):
        predictions = []
        for row in x_test:
            label = self.closest(row)
            predictions.append(label)

        return predictions

    def closest(self, row):
        best_dist = euclidean_distance(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euclidean_distance(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.Y_train[best_index]


myScrappyKNNClassifier = ScrappyKNN()
myScrappyKNNClassifier.fit(X_train, Y_train)
myScrappyKNNClassifierPredictions = myScrappyKNNClassifier.predict(X_test)
print myScrappyKNNClassifierPredictions

from sklearn.metrics import accuracy_score

print "accuracy_score"
print "decisionTreeClassifierPredictions"
print accuracy_score(Y_test, decisionTreeClassifierPredictions)
print "kNeighboursClassifierPredictions"
print accuracy_score(Y_test, kNeighboursClassifierPredictions)
print "myScrappyKNNClassifierPredictions"
print accuracy_score(Y_test, myScrappyKNNClassifierPredictions)
