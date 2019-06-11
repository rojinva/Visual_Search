# -*- coding: utf-8 -*-
"""
move files to create training sets
"""

import sys
import os
import shutil


f = open(sys.argv[1], "r")
lines = f.readlines()

for line in lines:
    toks = line.strip().split(',')
    fname = toks[0]+'.jpg'
    t = toks[1]
    src = os.path.join(os.getcwd(),'search_webui/static/images',fname)
    if t == 'solid':
        shutil.copy2(src,os.path.join('solid_images',fname))
    elif t == 'pattern':
        shutil.copy2(src,os.path.join('pattern_images',fname))
