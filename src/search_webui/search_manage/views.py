from django.shortcuts import render
from django.views.decorators.csrf import csrf_protect
from django.http import HttpResponse
from django.template import RequestContext, loader
from random import shuffle
import json
import cv2
import numpy

# Create your views here.
import os
import pickle
import random
import json
from operator import itemgetter
from src import ImageSim
from src.search_webui.manage import prj_name
filenames = os.listdir("static/images_%s"%(prj_name))
filenames = [f for f in filenames if '.jpg' in f]
lookup_contours = pickle.load(open("../../pickles/%s/index_contours.pkl"%prj_name,"r"))
lookup_hist = pickle.load(open("../../pickles/%s/index_hist.pkl"%prj_name,"r"))
lookup_gabor = pickle.load(open("../../pickles/%s/index_gabor.pkl"%prj_name,"r"))
lookup_lbp = pickle.load(open("../../pickles/%s/index_lbp.pkl"%prj_name,"r"))

h = ImageSim.Histogram()
c = ImageSim.Contours()
g = ImageSim.GaborPatterns()
l = ImageSim.LocalBinaryPatterns()


config = json.load(open("../config.json","r"))

def index(request):
    results = []
    shuffle(filenames)
    for fname in filenames[:config["num_random_images"]]:
        data = {}
        data['image_url'] = fname
        data['sku'] = fname.strip('.jpg')
        results.append(data)
    prj_name_ = prj_name
    template = loader.get_template('base_template.html')
    context = RequestContext(request, locals())
    return HttpResponse(template.render(context))

def label(request):
    sku = request.GET['sku']
    l = request.GET['l']
    fp = open("data","a")
    fp.write("%s,%s\n"%(sku,l))
    fp.close()
    return HttpResponse("ok")


def show_sift(request):
    sku = request.GET['sku']
    fname = os.path.join('static','images',sku+'.jpg')
    img = cv2.imread(fname)
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(imgray, 250, 255, cv2.THRESH_BINARY)
    sift = cv2.SIFT()
    masked = 255 - im_bw
    kp = sift.detect(imgray,masked)
    imgsift=cv2.drawKeypoints(imgray,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(os.path.join('static','mask.jpg'), imgsift)
    return HttpResponse(json.dumps([{'data':'bar'}]), content_type="application/json")

def show_debug(request):
    results = []
    sku = request.GET['sku']
    fname = os.path.join('static','images_%s'%prj_name,sku+'.jpg')
    img = ImageSim.Image(fname)
    randfname_mask = 'm_'+ str(random.randint(0,1000))+'.jpg'
    cv2.imwrite(os.path.join('static','temp',randfname_mask), img.mask)
    results.append({'fname':randfname_mask})

    #data   = h.hist_debug_hsv(img)
#    for d in data:
#        hist = d[0]
#        bins = d[1]
#        print hist
#        width = 0.7 * (bins[1] - bins[0])
#        center = (bins[:-1] + bins[1:]) / 2
#        plt.bar(center, hist, align='center', width=width)
#        fname = 'h_'+ str(random.randint(0,1000))+'.jpg'
#        filepath = os.path.join('static','temp',fname )
#        plt.savefig(filepath)
#        plt.close()
#        results.append({'fname':fname, 'sim_score':0})
    return HttpResponse(json.dumps({'data':results}), content_type="application/json")

def normalize(data):
    vals = [d[1] for d in data]
    max_val = max(vals)
    min_val = min(vals)
    range_val = max_val - min_val
    normalised_data = [(d[0],(d[1]-min_val)*1.0/range_val) for d in data]
    return normalised_data

def make_results(sims_sorted, is_display, k = config["num_matches"]):
    results = []
    if not is_display:
        return []
    for f,sim in sims_sorted[:k]:
        data = {}
        data['sku'] = f.strip('.jpg')
        data['image_url'] = f
        data['sim_score'] = "%.3f"%sim
        results.append(data)
    return results

def show_similar(request):
    sku_id = request.GET['sku']
    sku = sku_id + '.jpg'
    hist1 = lookup_hist[sku]
    cont1 = lookup_contours[sku]
    gabor1 = lookup_gabor[sku]
    lbp1 = lookup_lbp[sku]
    pattern_type = gabor1[0]
    gabor_vector = gabor1[2]
    lbp_vector = lbp1[1]
    sims1 = []
    sims2 = []
    sims3 = []
    sims4 = []
    for k,v in lookup_hist.items():
        s1 = h.similarity(hist1,v)
        s2 = c.intersect_area(cont1, lookup_contours[k])
        sims1.append((k,s1))
        sims2.append((k,s2))
        gabor = lookup_gabor[k]
        if pattern_type == gabor[0]:
            s3 = g.similarity(gabor_vector, gabor[2])
            sims3.append((k,s3))
        lbp = lookup_lbp[k]
        if pattern_type == gabor[0]:
            s4 = l.similarity(lbp_vector, lbp[1] )
            sims4.append((k,s4))
    normalised_sims3 = normalize(sims3)
    normalised_sims4 = normalize(sims4)
    sims5  = []
    for d1,d2 in zip(normalised_sims3, normalised_sims4):
        assert d1[0] == d2[0]
        k = d1[0]
        s = (d1[1] + d2[1])*0.5
        sims5.append((k,s))
    sims1_sorted = sorted(sims1, key = itemgetter(1))
    sims2_sorted = sorted(sims2, key = itemgetter(1), reverse=True)
    sims3_sorted = sorted(normalised_sims3, key = itemgetter(1))
    sims4_sorted = sorted(normalised_sims4, key = itemgetter(1))
    sims5_sorted = sorted(sims5, key = itemgetter(1))
    results1 = make_results(sims1_sorted, config["hist"])
    results2 = make_results(sims2_sorted, config["shape"])
    results3 = make_results(sims3_sorted, config["gabor"])
    results4 = make_results(sims4_sorted, config["lbp"])
    results5 = make_results(sims5_sorted, config["gabor_lbp"])
    similar_products = {'data1':results1, 'data2':results2, 'data3':results3, 'data4': results4, 'data5': results5, 'pattern_type':pattern_type, 'prj_name':prj_name}
    return HttpResponse(json.dumps(similar_products), content_type="application/json")