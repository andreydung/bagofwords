# Implement bag of words for images. 
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import sparse_encode
from sklearn.decomposition import MiniBatchDictionaryLearning

class BOW:
	sift = cv2.SIFT()
	
	def __init__(self, mypaths, myN=200):
		"""
		paths: a list of image paths
		N: number of codebook clusters, default to be 100
		"""
		self.paths = mypaths
		self.N_codebook = myN
		self.patchSize = 32
		self.gridSpacing = 8

		self.raw_features()
		self.codebook()

	def raw_feature_extract(self, path):
		im = cv2.imread(path)
		if (im==None):
			raise Exception, "Invalid image path %s" %path

		(M, N) = im.shape[:2]
		# Grid sampling
		pointlist=[]
		for i in range(1,M, self.patchSize):
			for j in range(1, N, self.patchSize):
				pointlist.append(cv2.KeyPoint(i,j,1))

		kps, des = self.sift.compute(im, pointlist)
		return des
			
	def raw_features(self):
		raw_feature=np.zeros((1,128))
		for path in self.paths:
			des = self.raw_feature_extract(path)
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

	
	def feature(self, path):
		des = self.raw_feature_extract(path)
		labels = self.est.predict(des)
		h, edge = np.histogram(labels,bins=np.array(range(self.codebook.shape[0]+1))-0.5,density=True)
		return np.asarray([h])


class BOW_spatialpyramid(BOW):
	def codebook(self):
		raw_feature = np.genfromtxt("raw_feature.csv", delimiter=',')
		mdbl =  MiniBatchDictionaryLearning(self.N_codebook)
		mbdl.fit(raw_feature)
		self.dictionary = mbdl

	def feature(self, path):
		des = self.raw_feature_extract(path)
		return sparse_encode(des, self.dictionary.components_)



