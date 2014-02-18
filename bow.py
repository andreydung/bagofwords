# Implement bag of words for images. 
import cv2
import numpy as np
from sklearn.cluster import KMeans

class BOW:
	def __init__(self, mypaths, mylabels, myN=200):
		"""
		paths: a list of image paths
		N: number of codebook clusters, default to be 100
		"""
		self.paths = mypaths
		self.N_codebook = myN
		self.patchSize = 16
		self.gridSpacing = 8
		self.labels = mylabels

	def raw_feature(self):
		sift = cv2.SIFT()
		for path in self.paths:
			im = cv2.imread(path)
			if (im==None):
				print path
				raise Exception, "Invalid image path %s" %path

			(M, N) = im.shape[:2]
			
			# Grid sampling
			raw_feature=np.zeros((1,128))
			pointlist=[]
			for i in range(1,M, self.patchSize):
				for j in range(1, N, self.patchSize):
					pointlist.append(cv2.KeyPoint(i,j,1))

			kps, des = sift.compute(im, pointlist)
			raw_feature=np.append(raw_feature,des,axis=0)

		raw_feature = np.delete(raw_feature, (0), axis=0)
		print "Raw feature shape: "+str(raw_feature.shape)
		np.savetxt("raw_feature.csv", raw_feature, fmt='%.4f', delimiter=',')

	def codebook(self):
		raw_feature = np.genfromtxt("raw_feature.csv", delimiter=',')

		est=KMeans(init='k-means++', n_clusters=self.N_codebook)
		est.fit(raw_feature)

		self.est=est
		self.codebook=est.cluster_centers_
		print "Codebook shape: "+str(self.codebook.shape)
		np.savetxt("codebook_kmeans.csv", self.codebook, fmt='%.3f', delimiter=',')

	
	def feature(self):
		sift=cv2.SIFT()
		kp, des=sift.detectAndCompute(im, mask)

		if len(kp):
			labels=self.est.predict(des)
			h, edge = np.histogram(labels,bins=np.array(range(self.codebook.shape[0]+1))-0.5,density=True)
			return np.asarray([h])

	def train():

		# try:
		# 	print "Loading saved codebook"
		# 	self.est=pickle.load(open("codebook.p","rb"))
		# 	self.codebook=self.est.cluster_centers_
		# 	self.N_codebook=self.codebook.shape[0]
		# 	print "Codebook shape: "+str(self.codebook.shape)

		# except IOError:
		# 	print "There is no saved codebook"
		# 	position=[None]
		# 	feature=np.zeros((1,128))
		# 	image_index=[None]

		# 	sift=cv2.SIFT()

		# 	for path in paths:
		# 		im=cv2.imread(path)
		# 		kp, des=sift.detectAndCompute(im, None)
				
		# 		feature=np.append(feature,des,axis=0)
		# 		position=position+kp

		# 	print "====== Building codebook for BoW =========="
		# 	print "Feature shape: "+str(feature.shape)
			
		# 	# Perform kmean clustering
		# 	est=KMeans(init='k-means++', n_clusters=N)
		# 	est.fit(feature)

		# 	self.N_codebook=N
		# 	self.est=est
		# 	self.codebook=est.cluster_centers_
		# 	print "Codebook shape: "+str(self.codebook.shape)
		# 	pickle.dump(self.est,open("codebook.p","wb"))

class BOW_kmeans(BOW):
	def codebook(self):
		raw_feature = np.genfromtxt("raw_feature.csv", delimiter=',')

		est=KMeans(init='k-means++', n_clusters=self.N_codebook)
		est.fit(raw_feature)

		self.est=est
		self.codebook=est.cluster_centers_
		print "Codebook shape: "+str(self.codebook.shape)
		np.savetxt("codebook_kmeans.csv", self.codebook, fmt='%.3f', delimiter=',')
		


