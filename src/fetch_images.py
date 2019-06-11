# -*- coding: utf-8 -*-
"""
script to download images
"""

import urllib2
import json
import sys
import os
skus_fetched = {}
infile = open(sys.argv[1], 'r')
for line in infile:
    doc = json.loads(line)
    image_url = doc['large_image_url']
    sku = doc["sku"]
    parent_sku = doc['parent_sku']
    if parent_sku != ' ':
        sku = parent_sku
    sku = sku.split('|')[0]
    if sku not in skus_fetched and 'SWIMWEAR' not in doc['category']:
        print image_url
        image = urllib2.urlopen(image_url)
        outf = open(os.path.join('/Users/z080465/images4',sku+".jpg"), 'wb')
        outf.write(image.read())
        outf.close()
        skus_fetched[sku] = 1