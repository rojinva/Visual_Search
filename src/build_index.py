# -*- coding: utf-8 -*-
"""
script to generate image features & store the vectors in a dict that is pickled.
Expects pre trained svm classifier for pattern
3 indexes built: color, shape, pattern
"""
from sklearn.svm import LinearSVC
from ImageSim import Histogram
from ImageSim import Contours
from ImageSim import Image
from ImageSim import GaborPatterns
from ImageSim import LocalBinaryPatterns
import os
import sys
import pickle
dir1 = sys.argv[1] #dir with input image files
dir2 = sys.argv[2] #dir where pickles will be saved

clf = pickle.load(open(os.path.join(dir2, "sv.pkl"),"rb"))
means = pickle.load(open(os.path.join(dir2, "means.pkl"), "rb"))
stddevs = pickle.load(open(os.path.join(dir2, "stddevs.pkl"), "rb"))

h = Histogram()
c = Contours()
g = GaborPatterns()
l = LocalBinaryPatterns()

files = os.listdir(dir1)
files = [f for f in files if '.jpg' in f]
index_hist = {}
index_contours = {}
index_gabor = {}
index_lbp = {}
for f in files:
    fname = os.path.join(dir1,f)
    print fname
    img = Image(fname)
    lbp = l.lbp_feature(img)
    hist = h.hist_hsv(img)
    assert len(hist) == 48
    index_hist[f] = hist
    hist = h.hist_bgr(img)
    cont = c.largest_contour(img)
    gf, gaborimages = g.gabor_features(img)
    features = []
    for idx,raw_f in enumerate(gf):
        features.append((raw_f-means[idx])*1.0/stddevs[idx])
    pred = clf.predict(features)
    if pred == [0]:
        t = 'solid'
    elif pred == [1]:
        t = 'pattern'
    print fname,t
    index_hist[f] = hist
    index_contours[f] = cont
    index_gabor[f] = (t,gf,features)
    index_lbp[f] = ('solid', lbp)

pickle.dump(index_hist, open(os.path.join(dir2, "index_hist.pkl"),"w"))
pickle.dump(index_contours, open(os.path.join(dir2,"index_contours.pkl"),"w"))
pickle.dump(index_gabor, open(os.path.join(dir2,"index_gabor.pkl"),"w"))
pickle.dump(index_lbp, open(os.path.join(dir2,"index_lbp.pkl"),"w"))

