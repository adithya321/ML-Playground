from sklearn import tree

'''
Features
1. Weight
2. Texture : 0 - Bumpy, 1 - Smooth
'''
features = [[140, 1], [130, 1], [150, 0], [170, 0]]

# Labels : 0 - Apple, 1 - Orange
labels = [0, 0, 1, 1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

print clf.predict([[160, 0]])
