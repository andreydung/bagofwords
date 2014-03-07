# IPSILON approach
import util
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

train = np.genfromtxt('../IPSILON/ipsilon_short_train.csv', delimiter = ',')
test = np.genfromtxt('../IPSILON/ipsilon_short_test.csv', delimiter = ',')

print train
print test

train_folder = '/media/sda5/Projects/Semantic/Database/Stanford/scene15_short'
(paths, labels, categories) = util.getpaths(train_folder)

test_folder = '/media/sda5/Projects/Semantic/Database/Stanford/scene15_test'
(testpaths, testlabels, testcategories) = util.getpaths(test_folder)

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(train, labels)
prediction = neigh.predict(test)

print prediction
print testlabels

print "Classfication accuracy: "
print float(sum(prediction==testlabels))/len(prediction)


