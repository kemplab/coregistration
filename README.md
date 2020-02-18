This code uses imzmlparser.py library copied from https://github.com/alexandrovteam/pyimzML/blob/master/pyimzml/ImzMLParser.py,
and mykmeans.py which is copied from https://github.com/scikit-learn/scikit-learn/blob/b194674c4/sklearn/cluster/_kmeans.py with all the euclidean distances changed to cosine distances.


Folders "confocal day5_1" and "maldi day5_1" are examples of corresponding inputs for the 
"coregistration pipeline no imzml.ipynb";
"metrics day 5_1", "cells day 5_1.xlsx", "cropped_confocal day5_1" and "maldi aligned day 5_1" are corresponding outputs.
There is no need to download them in order for the scripts to work.


1. Running "full coregistration pipeline.ipynb". Download "mykmeans.py","imzmlparser.py","coregistration_metrics.py" and "full coregistration pipeline.ipynb" and keep them in the same folder.

    Inputs for this script are: imzML file from the MALDI run and a folder with confocal images. You can adjust all the paths and parameters in the very first cell of the "full coregistration pipeline.ipynb" Jupyter Notebook.

    Outputs from the script are: folder with ion images, folder with aligned ion images, folder with cropped confocal images, folder with metric images, .xslx file with cell-by-cell intensity data, .csv file with cell-by-cell metrics data.
    
2. In case you already have your ion images created by some other software you can run a "coregistration pipeline no imzml.ipynb".

    This script only uses our custom library "coregistration_metrics.py", so keep it in the same folder as "coregistration pipeline no imzml.ipynb".
    
    Inputs for this script are: a folder with ion images and a folder with confocal images. You can adjust all the paths and parameters in the very first cell of the "coregistration pipeline no imzml.ipynb" Jupyter Notebook.
    
    Outputs from the script are the same as for #1, except for the folder with ion images.
    
3. To create ion images separately run "ion_maker.py". It uses "mykmeans.py" and "imzmlparser.py", so keep them in the same folder as the "ion_maker.py".

    Input: imzML file from the MALDI run
    
    Output: a folder with ion images
    
The code uses Python 3.7.6 and conda 4.8.2 with opencv 3.4.2, scikit-image 0.16.2, scikit-learn 0.22.1, scipy 1.4.1, numpy 1.18.1 

For any issues with the code write to anikitina3@gatech.edu
