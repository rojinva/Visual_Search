Visual Similarity
=================

This demo calculates the following types of visual similarity for apparel. 

1. Color Histogram 
2. Shape Similarity
3. Pattern Similarity (two methods) 
   1. Gabor features
   2. Local Binary patterns

Pattern similarity is calculated after classifying the pattern into either *solid* or *pattern* using a Linear SVM classifier. 
The demo has 926 images for women's apparel & 1193 images for footwear.

###Dependencies
1. opencv (mac users https://jjyap.wordpress.com/2014/05/24/installing-opencv-2-4-9-on-mac-osx-with-python-support/ )
2. scikit-image (http://scikit-image.org/download.html)
3. django (https://www.djangoproject.com/download)
4. If you want to build the index for every image ,you also need scikit-learn &  numpy 

###Running the demo
To start the demo navigate to `visualsearch/src/search_webui` and start the webapp with this command

`python manage.py runserver 8010 apparel` *Runs the webapp locally on port 8010 for apparel*

Similarly if you want to run the demo for footwear

`python manage.py runserver 8011 footwear` 

Some things to know about the UI 

1. Refreshing the page shows a random set of 20 images
2. Clicking on any image opens a popup that has three rows, each for a kind of similarity.
    1. The leftmost image is the query image (the one clicked)
    2. Only way to close the popup is to click on the X button in the top right
    3. The images in one row might overflow into another row depending on the width of the page :)
3. By default three kinds of similarity are shown for every image: Color, Shape & Pattern(blend of gabor & LBP)
4. To change defaults (number of random images, number of matches, kinds of similarities) edit `src/config.json` and restart webapp

#####Footwear screenshot
![alt tag](https://git.target.com/z080465/visualsearch/raw/master/footwear_screenshot.png)
#####Apparel screenshot
![alt tag](https://git.target.com/z080465/visualsearch/raw/master/apparel_screenshot.png)
##Questions 
satyajit.gupte@target.com
