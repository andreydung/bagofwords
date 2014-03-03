import os
import numpy as np

# Get labels from subdirectories
def getpaths(root):
	"""
	Each category in one sub folder
	"""
	categories = os.listdir(root)
	for folder in categories:
		if not os.path.isdir(os.path.join(root, folder)):
			raise Exception("Invalid database. One folder for each category")
	
	print "Categories:"
	print categories

	paths = []
	labels = []

	for i in range(len(categories)):
		count = 0
		for f in os.listdir(os.path.join(root, categories[i])):
			if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".tiff"):
				count +=1
				paths.append(os.path.join(root, categories[i], f))
				labels.append(i)
		print "     %d images in category \"%s\"" % (count, categories[i])

	labels=np.array(labels)
	print "Total %s images" % len(paths)
	
	return paths, labels, categories



if __name__ =="__main__":
	folder = '/media/sda5/Projects/Semantic/Database/Stanford/Fulltest/scene15_large_test'
	(paths, labels, categories) = getpaths(folder)

	thefile = open('scene15_short.txt','w')

	for path in paths:
		thefile.write("%s\n" % path)
