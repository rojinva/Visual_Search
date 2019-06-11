# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 21:43:33 2014

@author: z080465
"""
from __future__ import division
import cv2
import numpy as np
from skimage.feature import local_binary_pattern

class Image:
    def __init__(self, file_path):
        self.bgr = cv2.imread(file_path)
        self.gray = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY)
        self.hsv = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2HSV)
        thresh, bw = cv2.threshold(self.gray, 250, 255, cv2.THRESH_BINARY)
        self.bw = bw
        self.mask = 255 - bw


class Histogram:
    """ Compute Histograms in HSV space.
    Pixels having saturation less than 15 are treated as Grey pixels
    """
    def __init__(self, n_bins = [24,8,8,8], ranges = [(0,180), (0,256), (0,256), (0,256), (0,15)], weights = [0.4, 0.2, 0.1, 0.3]):
        """n_bins, ranges, weights for for H, S, V, G"""
        self.n_bins = n_bins
        self.ranges = ranges
        self.weights = weights

    def hist_bgr(self, img):
        hist = cv2.calcHist([img.hsv], [0,1,2], img.mask, [32,64,2], [0,180,0,256,0,256])
        return hist.flatten()

    def split_hsv_channels(self, img):
        mask_indices = np.nonzero(img.mask.ravel())
        hsv_channels = []
        for idx in range(3):
            channel = img.hsv[:,:,idx]
            channel = np.take(channel, mask_indices)
            hsv_channels.append(channel)
        return hsv_channels

    def filter_channels(self, channels, indices):
        """delete pixels at indices from channel"""
        filtered_channels = []
        for channel in channels:
            c = np.delete(channel, indices)
            filtered_channels.append(c)
        return filtered_channels

    def hsvg_channel_histograms(self, hsvg_channels, num_grey_pixels):
        """compute weighted hsvg histogams. Bins normalised by number of grey pixels """
        histograms = []
        for idx, channel in enumerate(hsvg_channels):
            h, b = np.histogram(channel, self.n_bins[idx], self.ranges[idx])
            h = h*self.weights[idx]/num_grey_pixels
            histograms.append(h)
        return histograms

    def hist_hsv(self, img):
        hsv_channels = self.split_hsv_channels(img)
        s = hsv_channels[1]
        grey_indices = np.nonzero(s<15) # grey pixels have saturation < 15
        num_grey_pixels = len(grey_indices[0])
        g = np.take(s, grey_indices)
        #filtered_channels = self.filter_channels(hsv_channels, grey_indices)
        hsv_channels.append(g) # add grey channel
        hsvg_histograms = self.hsvg_channel_histograms(hsv_channels, num_grey_pixels)
        hist = np.hstack(hsvg_histograms)
        return hist.astype('float32')

    def similarity(self, hist1, hist2):
        return cv2.compareHist(hist1, hist2, cv2.cv.CV_COMP_BHATTACHARYYA)

    def hist_debug_hsv(self, img):
        data = []
        mask_indices = np.nonzero(img.mask.ravel())
        h_, s_, v_ = cv2.split(img.hsv)
        h = np.take(h_, mask_indices)
        s = np.take(s_, mask_indices)
        v = np.take(v_, mask_indices)
        grey_indices = np.nonzero(s<15) # grey pixels have lesser saturation
        num_grey_pixels = len(grey_indices[0])
        g = np.take(s, grey_indices)
        h_hist, h_bins = np.histogram(h, 24, (0,180))
        s_hist, s_bins = np.histogram(s, 8, (0,256))
        v_hist, v_bins = np.histogram(v, 8, (0,256))
        g_hist, g_bins = np.histogram(g, 8, (0,15))
        g_hist = g_hist*0.3/num_grey_pixels
        h1 = np.delete(h, grey_indices)
        h1_hist, h1_bins = np.histogram(h1, 24, (0,180))
        h1_hist = h1_hist*0.4/num_grey_pixels
        s1 = np.delete(s, grey_indices)
        s1_hist, s1_bins = np.histogram(s1, 8, (0,256))
        s1_hist = s1_hist*0.2/num_grey_pixels
        v1 = np.delete(v, grey_indices)
        v1_hist, v1_bins = np.histogram(v1, 8, (0,256))
        v1_hist = v1_hist*0.1/num_grey_pixels
        data.append((h_hist, h_bins))
        data.append((s_hist, s_bins))
        data.append((v_hist, v_bins))
        data.append((g_hist, g_bins))
        data.append((h1_hist, h1_bins))
        data.append((s1_hist, s1_bins))
        data.append((v1_hist, v1_bins))
        hist = np.hstack((h1_hist, s1_hist, v1_hist, g_hist))
        #return hist.astype('float32')
        return data

class Contours:
    def largest_contour(self, img):
        contours, hierarchy = cv2.findContours(img.bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)
        return sorted_contours[1] # hacky --> 0th contour is the image border !

    def similarity(self, contour1, contour2):
        return cv2.matchShapes(contour1, contour2, cv2.cv.CV_CONTOURS_MATCH_I3, 0)

    def intersect_area(self, contour1, contour2):
        blank1 = np.zeros((410,410))
        blank2 = np.zeros((410,410) )
        filled1 = cv2.drawContours(blank1,[contour1],0,255,-1)
        filled2 = cv2.drawContours(blank2,[contour2],0,255,-1)
        intersection = cv2.bitwise_and(blank1, blank2)
        intersect_area = cv2.countNonZero(intersection)
        union = cv2.bitwise_or(blank1, blank2)
        union_area = cv2.countNonZero(union)
        return intersect_area*1.0/union_area


class GaborPatterns:
    def __init__(self, lambdas=[2.3,3.0,4.0,5.0,6.0], gammas=[0.10,0.20,0.30,0.5,1.0]):
        self.filters = []
        ksize = 31 #kernel size
        for lambd in lambdas:
            for theta in np.arange(0, np.pi, np.pi / 4):
                for g in gammas:
                    params = {'ksize':(ksize, ksize), 'sigma':0.56*lambd, 'theta':theta, 'lambd':lambd,
                             'gamma':g, 'psi':0, 'ktype':cv2.CV_32F}
                    kern = cv2.getGaborKernel(**params)
                    kern /= 1.5*kern.sum() #divide the weighted sum to scale down intensity of convolved pixel
                    self.filters.append((kern,params))

    def gabor_filter(self, img):
        i = cv2.bitwise_and(img.gray, img.mask)
        results = []
        for kern,params in self.filters:
            fimg = cv2.filter2D(img.gray, cv2.CV_8UC3, kern)
            results.append(fimg)
        return results

    def gabor_features(self, img):
        features = []
        thres_images = []
        gabor_images = self.gabor_filter(img)
        for gabor_filtered_image in gabor_images:
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(gabor_filtered_image, img.mask)
            thresh, bw = cv2.threshold(gabor_filtered_image, maxVal-2, 255, cv2.THRESH_BINARY)
            mean =  cv2.mean(bw, img.mask)[0]
            features.append(mean)
            thres_images.append(bw)
        return features,thres_images

    def similarity(self, f1, f2):
        return np.linalg.norm(np.array(f1)-np.array(f2))

class LocalBinaryPatterns:
    def lbp_feature(self, img, rads = [(1,8),(2,16),(3,24),(4,16)]):
        features = []
        mask_indices = np.nonzero(img.mask.ravel())
        for rad in rads:
            r = rad[0]
            n = rad[1]
            lbp = local_binary_pattern(img.gray, n, r, 'uniform')
            lbp_flat = lbp.ravel()
            lbp_masked = np.take(lbp_flat, mask_indices)
            n_bins = lbp_masked.max() + 1
            hist, _ = np.histogram(lbp_masked, normed=True, bins=n_bins, range=(0, n_bins))
            features.append(hist)
        return features

    def similarity(self, f1, f2):
        s = []
        for h1, h2 in zip(f1,f2):
            distance = np.linalg.norm(np.array(h1)-np.array(h2))
            s.append(distance)
        return np.mean(s)



#
# testing
#

if __name__ == '__main__':
    fname1 = '' #large stripes
    #fname2 = '' #red sweater
    fname2 = '' #green skirt
    img1 = Image(fname1)
    img2 = Image(fname2)
    h = Histogram()
    hist = h.hist_hsv(img2)
    #l = LocalBinaryPatterns()
    #h1 = l.lbp_feature(img1)
    #h2 = l.lbp_feature(img2)
    #print l.similarity(h1, h2)