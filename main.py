import os
from bow import *
from bow_classifier import *

folder = '/media/sda5/Projects/Semantic/Database/Stanford/scene15_short'

# Get labels from subdirectories

categories = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder,name))]
print categories

cat_dict=dict()
for i in range(len(categories)):
	cat_dict[categories[i]]=i

print cat_dict

paths = []
labels = []
for path, subdirs, files in os.walk(folder):
	for name in files:
		paths.append(os.path.join(path, name))
		labels.append(cat_dict[os.path.split(path)[-1]])
print labels

b = BOW(paths, labels, 10)
b.raw_feature()
b.codebook()



# c = BOW_classifier('codebook_kmeans.csv', 'raw_feature.csv')