# -*- coding: utf-8 -*-
"""
utility script to test SVM classifier
"""

import pickle
import os
import sys
import shutil
from sklearn.svm import LinearSVC
from ImageSim import Image
from ImageSim import GaborPatterns

g = GaborPatterns()
files = os.listdir(sys.argv[1])
files = [f for f in files if '.jpg' in f]

clf = pickle.load(open("sv.pkl", "rb"))

for f in files:
    fname = os.path.join(sys.argv[1],f)
    i = Image(fname)
    features, _ = g.gabor_features(i)
    y = clf.predict(features)
    if y == [0]:
        shutil.copy2(fname, os.path.join('label_solid',f))
    elif y == [1]:
        shutil.copy2(fname, os.path.join('label_pattern',f))