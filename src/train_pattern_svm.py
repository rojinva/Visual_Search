# -*- coding: utf-8 -*-
"""
learn svm classifier for pattern
"""
from __future__ import division
import os
import sys
from sklearn.svm import LinearSVC
from random import shuffle
from collections import defaultdict
import pickle
import numpy

dirs = ('solid_images', 'pattern_images')


from ImageSim import Image
from ImageSim import GaborPatterns
import math
raw_corpus = []
g = GaborPatterns()
for idx, d in enumerate(dirs):
    files = os.listdir(d)
    files = [os.path.join(os.getcwd(),d,f) for f in files if '.jpg' in f]
    for f in files:
        img = Image(f)
        raw_features, images = g.gabor_features(img)
        #length = math.sqrt(sum(f**2 for f in raw_features))
        #features = [f*1.0/length for f in raw_features]
        raw_corpus.append((raw_features,idx))

means = []
stddevs = []
num_features = len(raw_corpus[0][0])
for f_idx in range(0,num_features):
    row = [row[0][f_idx] for row in raw_corpus]
    f_mean = numpy.mean(row)
    f_stddev = numpy.std(row)
    means.append(f_mean)
    stddevs.append(f_stddev)

corpus = []
for raw_row in raw_corpus:
    raw_features = raw_row[0]
    features  = []
    for idx,f in enumerate(raw_features):
        features.append((f-means[idx])/stddevs[idx])
    corpus.append((features, raw_row[1]))

shuffle(corpus)
cut = int(len(corpus)*0.8)
x_train = []
y_train = []
for data in corpus[:cut]:
    x_train.append(data[0])
    y_train.append(data[1])
x_test = []
y_test = []
for data in corpus[cut:]:
    x_test.append(data[0])
    y_test.append(data[1])
print len(x_test), len(y_test), len(x_train), len(y_train)

clf = LinearSVC()
clf.fit(x_train, y_train)
print "done fit"
pred = clf.predict(x_test)

total = 0
correct = 0
classif = defaultdict(int)
for i in range(len(pred)):
    classif[(y_test[i],pred[i])]+=1
    if pred[i] == y_test[i]:
        correct+=1
    total+=1

print correct*1.0/total
print correct
print total
print classif
pickle.dump(clf, open("sv.pkl","wb"))
pickle.dump(means, open('means.pkl', 'wb'))
pickle.dump(stddevs, open('stddevs.pkl', 'wb'))
