import os
import numpy as np

# Get labels from subdirectories
def getpaths(folder):
	categories = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder,name))]
	print "Categories:"
	print categories

	cat_dict=dict()
	for i in range(len(categories)):
		cat_dict[categories[i]]=i

	paths = []
	labels = []
	for path, subdirs, files in os.walk(folder):
		for name in files:
			paths.append(os.path.join(path, name))
			labels.append(cat_dict[os.path.split(path)[-1]])

	labels=np.array(labels)
	print "There are totally %s images" % len(path)
	return paths, labels, categories


# paths_train, paths_test, y_train, y_test = cross_validation.train_test_split(paths, labels,\
# 														 test_size=0.3, random_state=0)

# paths_train=paths_train.tolist()
# paths_test=paths_test.tolist()
# y_train=y_train.tolist()
# y_test=y_test.tolist()

# b = BOW(paths_train, y_train, 100)
# b.train()


# prediction = b.test(paths_test)
# true = np.array(y_test)

# print prediction
# print true

# print sum(prediction!=true)




# c = BOW_classifier('codebook_kmeans.csv', 'raw_feature.csv')