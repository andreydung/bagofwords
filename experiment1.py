from bow import *
import util
from sklearn import cross_validation
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

train_folder = '/media/sda5/Projects/Semantic/Database/Stanford/scene15_short'
(paths, labels, categories) = util.getpaths(train_folder)

b = BOW(paths)

X = b.feature(paths)

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X, labels)

test_folder = '/media/sda5/Projects/Semantic/Database/Stanford/scene15_test'
(testpaths, testlabels, testcategories) = util.getpaths(test_folder)

X_test = b.feature(testpaths)

prediction = neigh.predict(X_test)

print prediction
print testlabels

print "Classfication accuracy: "
print float(sum(prediction==testlabels))/len(prediction)

