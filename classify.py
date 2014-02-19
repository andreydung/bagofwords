from sklearn import svm
from bow import *
import util
from sklearn import cross_validation

folder = '/media/sda5/Projects/Semantic/Database/Stanford/scene15_short'

(paths, labels, categories) = util.getpaths(folder)

b = BOW(paths)

X = np.zeros((1,self.N_codebook))
for path in paths:
	des = b.feature(path)
	X = np.append(X, des, axis=0)
X = np.delete(X, (0), axis=0)

clf = svm.SVC()
scores = cross_validation.cross_val_score(clf, X, labels, cv =5)
