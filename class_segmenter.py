#Keras update imports
import os
import fileinput

# Define the path to the file that needs to be modified
FILE_PATH = "C:\\Users\\Scott Moran\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python310\\site-packages\\pixellib\\semantic\\deeplab.py"

# Define the old and new strings that need to be replaced
OLD_STRING = "tensorflow.python.keras"
NEW_STRING = "tensorflow.keras"

# Use fileinput to replace the old string with the new string in the file
for line in fileinput.input(FILE_PATH, inplace=True):
    print(line.replace(OLD_STRING, NEW_STRING), end='')

# Define the old and new strings that need to be replaced
# This handles model loading errors
OLD_STRING = "tensorflow.keras.utils.layer_utils import get_source_inputs"
NEW_STRING = "tensorflow.python.keras.utils.layer_utils import get_source_inputs"

# Use fileinput to replace the old string with the new string in the file
for line in fileinput.input(FILE_PATH, inplace=True):
    print(line.replace(OLD_STRING, NEW_STRING), end='')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
##################################################################

import glob
import pixellib
from pixellib.semantic import semantic_segmentation
import segment
import cv2
import numpy as np
import sys

import calendar
import time
#load model
segment_image = semantic_segmentation()
segment_image.load_ade20k_model("deeplabv3_xception65_ade20k.h5")

#get image directory
img_directory = 'C:\\Users\\Scott Moran\\Documents\\Research\\NMSProject-master\\PhantomSponges\\BDD_Dir\\BDD_IMG_DIR'

out_directory = 'C:\\Users\\Scott Moran\\Documents\\Research\\NMSProject-master\\out_segments'

current_out_dir = out_directory + '\\' + str(calendar.timegm(time.gmtime()))
#create the directory
os.mkdir(current_out_dir)

#loop through
print("Out folder: " + current_out_dir.split("\\")[-1])
idx_counter = 0
for impath in glob.iglob(img_directory + '\\*'):
    return_masks, return_ims = segment.segmentation_mask(segment_image, impath)
    idx_counter += 1
    print("Step " + str(idx_counter) + ": "  + ''.join(impath.split("\\")[-1].split('.')[:-1]))
    #Loop through each returned value and save accordingly
    for cls in return_masks:
        #Make directory if we need
        cls_folder = current_out_dir + '\\' + cls + '\\'
        if(len(return_masks[cls]) > 0) and not os.path.exists(cls_folder):
            os.mkdir(cls_folder)
        for i in range(0, len(return_masks[cls])):
            #Save image
            cv2.imwrite(cls_folder + ''.join(impath.split("\\")[-1].split('.')[:-1]) + '_' + str(i) + '.jpg', return_ims[cls][i])
            #save npz
            np.save(cls_folder + ''.join(impath.split("\\")[-1].split('.')[:-1]) + '_' + str(i) + '.npy', return_masks[cls][i])    