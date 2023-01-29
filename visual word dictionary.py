import numpy as np
import pandas as pd
import cv2 as cv
from os import listdir
from sklearn.cluster import KMeans
import joblib
import matplotlib.pyplot as plt


# load images into a numpy array
images = []
base = 'D:\\dip\\dataset\\'
datasetpath = 'D:\\dip\\dataset'
imglist = listdir(datasetpath)
basedirectory = 'D:\\dip\\dataset\\'
for i in imglist:
    img = cv.imread(base + i,0)
    images.append(img)
images = np.array(images)


# extract sift descriptors from images
# limited the number of descriptors to 250 of top descriptors with highest contrast in order to prevent large dataset 
extractor = cv.xfeatures2d.SIFT_create(nfeatures=250)
keypoints = []
descriptors = []
q = 0
for img in images:
    q+=1
    if q %100 == 0:
        print(q)
    img_keypoints, img_descriptors = extractor.detectAndCompute(img, None)
    if type(img_descriptors) == np.ndarray:
        keypoints.append(img_keypoints)
        descriptors.append(img_descriptors)
    else:
        descriptors.append(np.array([]))
        
        

# make all descriptors extracted from images into one array
alldescriptors = []
q = 0
for mat in descriptors:
    q+=1
    if q %100 == 0:
        print(q)
    for d in mat:
        alldescriptors.append(d)
alldescriptors = np.stack(alldescriptors)

# build a kmeans model to create a vocabulary of visual words with 250 visual words in it
savingpath = 'D:\\model6.sav'
model = KMeans(n_clusters=250,verbose=1,n_init=1)
model.fit(alldescriptors)
joblib.dump(model,savingpath)


# histogram for each image
features = []
q = 0
for i in descriptors:
    q+=1
    if q %100 == 0:
        print(q)
    hist = np.zeros(250)
    for j in i:
        x = model.predict(j.reshape(1,-1))
        hist[x[0]] += 1
    features.append(hist)
# term frequency of each image
features = np.array(features)
N = len(descriptors)
df = np.sum(features > 0, axis=0)
# tf-idf for each image
idf = np.log(N/ df)
df = pd.DataFrame(idf)
df.to_csv('D:\\idf.csv', index=False) 
tfidf = features * idf
df = pd.DataFrame(tfidf)
df.to_csv('D:\\tfidf.csv', index=False) 