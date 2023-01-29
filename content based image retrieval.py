import numpy as np
import pandas as pd
import cv2 as cv
from os import listdir
import joblib
import matplotlib.pyplot as plt


# postprocessing the initial results from retrieval with tf-idf
# re-ranking retrieved images based on global color histograms
def postprocess(test,rank,imglist,dist):
    feat = np.zeros(24)
    for im in test:
        testimg = cv.imread(im)
        testimg = cv.cvtColor(testimg,cv.COLOR_BGR2HSV)
        red = np.histogram(testimg[:,:,2],bins=8,range=(0,256))[0]
        green = np.histogram(testimg[:,:,1],bins=8,range=(0,256))[0]
        blue = np.histogram(testimg[:,:,0],bins=8,range=(0,256))[0]
        testhist = np.concatenate((red,green,blue))
        testhist = testhist / (480*640)
        testhist = np.array(testhist)
        feat = feat + testhist
    
    hists = []
    for i in rank:
        img = cv.imread(base+imglist[i])
        img = cv.cvtColor(img,cv.COLOR_BGR2HSV)
        red = np.histogram(img[:,:,2],bins=8,range=(0,256))[0]
        green = np.histogram(img[:,:,1],bins=8,range=(0,256))[0]
        blue = np.histogram(img[:,:,0],bins=8,range=(0,256))[0]
        hist = np.concatenate((red,green,blue))
        hist = hist / (480*640)
        hists.append(hist)
    hists = np.array(hists)
    cosine = np.dot(feat, hists.T)/(np.linalg.norm(feat) * np.linalg.norm(hists, axis=1))
    for i in range(len(cosine)):
        cosine[i] = cosine[i] + dist[rank[i]]
    idx = np.argsort(-cosine)
    best = []
    for i in idx:
        best.append(rank[i])
    
    return best[:20]




model = joblib.load('D:\\model6.sav')
tfidf = pd.read_csv('D:\\tfidf.csv')
idf = pd.read_csv('D:\\idf.csv')
idf = np.array(idf)
base = 'D:\\dip\\dataset\\'
base2 = 'D:\\test\\'
inputimage = input('please enter your image file name.')
test = [base2 + inputimage]
testimage = cv.imread(test[0],0)
im = cv.imread(test[0])
height = 3
width = 6
datasetpath = 'D:\\dip\\dataset'
imglist = listdir(datasetpath)


# testing on input image with relevance feedback
extractor = cv.xfeatures2d.SIFT_create(nfeatures=250)
while True:
    feat = np.zeros(250)
    q = 0
    if len(test) > 3:
        height = 5
        width = 10
    figure, axes = plt.subplots(nrows = 1, ncols = len(test),figsize=(width,height))
    for m in test:
        testimage = cv.imread(m,0)
        testim = cv.imread(m)
        testim = cv.cvtColor(testim,cv.COLOR_BGR2RGB)
        if len(test) == 1:
            axes.imshow(testim)
            axes.tick_params(axis = 'x', bottom=False, labelbottom=False,which='both',top=False)
            axes.tick_params(axis = 'y', left=False, labelleft=False,which='both')
        else:
            axes[q].imshow(testim)
            axes[q].tick_params(axis = 'x', bottom=False, labelbottom=False,which='both',top=False)
            axes[q].tick_params(axis = 'y', left=False, labelleft=False,which='both')
        q += 1
        figure.suptitle(' query ')
        keypoints, descriptors = extractor.detectAndCompute(testimage, None)
        hist = np.zeros(250)
        for i in descriptors:
            x = model.predict(i.reshape(1,-1))
            hist[x[0]] += 1
        hist = hist.reshape(1,250)
        idf = idf.reshape(1,250)
        hist = hist * idf
        feat = feat + hist
    plt.show()
    feat = feat / len(test)
    cosine = np.dot(feat, tfidf.T)/(np.linalg.norm(feat) * np.linalg.norm(tfidf, axis=1))
    idx = np.argsort(-cosine[0])[:100]
    idx = postprocess(test,idx,imglist,cosine[0])
    imglist = listdir('D:\\dip\\dataset')
    base = 'D:\\dip\\dataset\\'
    retrieved = []
    index = 1
    col = 0
    row = 0
    fig,ax = plt.subplots(nrows=2,ncols=10,figsize=(10,5),dpi=200)
    for i in idx:
        retrieved.append(imglist[i])
        img = cv.imread(base + imglist[i])
        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        ax[row][col].imshow(img)
        ax[row][col].set_title(index)
        ax[row][col].tick_params(axis = 'x', bottom=False, labelbottom=False)
        ax[row][col].tick_params(axis = 'y', left=False, labelleft=False)
        col += 1
        if col == 10:
            col = 0
        if index > 9 and row == 0:
            row += 1
        index += 1
    fig.subplots_adjust(hspace=0)
    plt.show()
    feedback = input('To continue the search with another image or images, enter their number seperated with ","\n else enter 0 to stop the retrieval.\n')
    if feedback == '0':
        break
    else:
        feeds = feedback.split(',')
        for f in feeds:
            f = int(f)
            test.append(base + retrieved[f-1])