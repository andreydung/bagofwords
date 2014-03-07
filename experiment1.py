# KMeans approach

from bow import *
import util
from sklearn import cross_validation
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

# train_folder = '/media/sda5/Projects/Semantic/Database/Stanford/scene15_short'
train_folder = '/media/sda5/Projects/Semantic/Database/Stanford/Fulltest/scene15'
(paths, labels, categories) = util.getpaths(train_folder)

b = BOW(paths, myN = 10)

X = b.feature(paths)

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X, labels)

# test_folder = '/media/sda5/Projects/Semantic/Database/Stanford/scene15_test'
test_folder = '/media/sda5/Projects/Semantic/Database/Stanford/Fulltest/scene15_large_test'
(testpaths, testlabels, testcategories) = util.getpaths(test_folder)

assert (categories == testcategories)

X_test = b.feature(testpaths)

prediction = neigh.predict(X_test)

print prediction
print testlabels

print "Classfication accuracy: "
print float(sum(prediction==testlabels))/len(prediction)

