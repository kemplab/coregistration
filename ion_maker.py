import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from skimage.filters import median, threshold_minimum, threshold_triangle
from skimage.morphology import disk
from coregistration_metrics import record_reader,getmaxmin,correlation_segmentation_reference,kmeanscorrelationfull
from imzmlparser import ImzMLParser
# This code provides you with ion images given a MALDI file in imzML format
# m/z values to get images of
ions =[253.131,281.275,303.247,616.472,716.551,724.538,768.575,778.528,835.518,883.576,885.573,887.57,913.597,915.594,940.592,1050.71]
# peak widths, values get summed from ion-tolerance to ion+tolerance
tolerances = [0.3]*len(ions)
# output folder for ion images
output_folder = "day5_1"
if not os.path.exists(output_folder): os.mkdir(output_folder)
# path to imzML MALDI file
MALDI_path = 'W://Arina//Li Li-Stem Cell Imaging//20180411_day5-8//day5/20180411_day5_box1//20180411_day5_box1.imzML'
p = ImzMLParser(MALDI_path)
k = 2
xmax, xmin, ymax, ymin = getmaxmin(p)
print(xmax, xmin, ymax, ymin)
record_reader([xmin,ymin,ymax,xmax],p,output_folder,ions,tolerances)

# get colony outlines from an average ion image
img = cv2.imread(output_folder+"/average.png",0)
t = threshold_minimum(img)
colony = img > t
S = (xmax-xmin)*(ymax-ymin)
if np.sum(colony) < S/10: 
        t = threshold_triangle(img)
        colony = img > t    
colony = median(colony, disk(5))
plt.imshow(colony)
plt.show()

# segment the colony into k regions with similar m/z spectra
# currently a k-means clustering of only the on-colony region is produced
# to see a whole image correlation remove "all = False"
kmeanscorrelationfull(p,colony,xmin,ymin,xmax,ymax,k,output_folder,all = False)
# you will get a text output "kmeans k output.txt" with coordinates and corresponding labels and a corresponding image

# a heatmap of correlation with a reference pixel spectrum:
# if you want to indicate a reference pixel add parameters x_ref = ... and y_ref = ... to the function call
# the coordinates do not start from zero, see xmax, xmin, ymax, ymin printed above
# if not indicated a random on-colony pixel is taken as reference (which is fine, you will still get a nice picture)
# currently a correlation heatmap of a whole image is produced, to only see the on-colony correlation include all = False in the function call
correlation_segmentation_reference(p,colony,xmin,ymin,xmax,ymax,output_folder)