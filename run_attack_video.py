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
from natsort import natsorted

import calendar
import time
import random
from PIL import Image

sys.path.append('C:\\Users\\Scott Moran\\Documents\\Research\\NMSProject-master\\CustomizedPhantomSponges')

from batchattack import run_attack

#load model
segment_image = semantic_segmentation()
segment_image.load_ade20k_model("deeplabv3_xception65_ade20k.h5")

#DIGITAL vs REAL WORLD switch
digital_mode = True

#get patch directory
patch_directory = 'D:\\Research\\phantom_sponge_out'

#output directory for digital experiments
out_directory = 'D:\\Research\\digital_videos'

#Get video feed for digital
video_directory = 'D:\\Research\\bdd100k_vids\\videos\\val'

current_out_dir = out_directory + '\\' + str(calendar.timegm(time.gmtime()))
#create the directory
os.mkdir(current_out_dir)

run_dictionary = True #Run dictionary attack
run_rt_generation = False #Run realtime generation of noise

if run_dictionary:
    os.mkdir(current_out_dir + '\\dict')
    
if run_rt_generation:
    os.mkdir(current_out_dir + '\\rw')
    
    
    
#Pick highest image iteration
def pick_highest_iter(patch_paths):
    highest_img = ''
    highest_num = -1
    for path in patch_paths:
        comparator = int(path.split("=")[-1].split(".")[0])
        if comparator > highest_num:
            highest_num = comparator
            highest_img = path
    return highest_img

def image_to_noise(img, target_size, return_mask_use=None, az_util_use=None):
    noise_mult = 2 # How much stronger the noise should be
    if return_mask_use is None or az_util_use is None:
        return_masks, return_ims, az_util = segment.segmentation_mask_im(segment_image, img)
    else:
        return_masks = return_mask_use
        az_util = az_util_use
    final_image = None
    #Loop through each returned value and pull the corresponding patch
    for cls in return_masks:
        #Find relevant patches
        patch_dir_1 = patch_directory + '\\' + cls + '*\\'
        #This one finds the ones under Qualcomm naming scheme
        patch_dir_2 = patch_directory + '\\results*' + cls + '*\\'
        considered_dirs = []
        if len(glob.glob(patch_dir_1)) > 0:
            considered_dirs.append(patch_dir_1)
        if len(glob.glob(patch_dir_2)) > 0:
            considered_dirs.append(patch_dir_2)
        if len(considered_dirs) <= 0:
            print("Unable to find class " + cls)
            continue
        #Pull together the patches and multiply the mask
        #Load a relevant patch at random
        patch_choice = pick_highest_iter(glob.glob(random.choice(glob.glob(random.choice(considered_dirs))) + 'save_patch\\*'))
        
        #Add to the current patch image
        current_patch = np.asarray(Image.open(patch_choice))
        rdim = (target_size[1], target_size[0])
        current_patch = cv2.resize(current_patch, rdim, interpolation=cv2.INTER_CUBIC)
        mask3d = np.stack((return_masks[cls][0],)*3, axis=-1)
        #Old interpolation code, commenting in case relevant
#        rdim = (len(mask_i[0]), len(mask_i))
#        transf = cv2.resize(transf, rdim, interpolation=cv2.INTER_CUBIC)
#        transf = np.multiply(transf, mask3d)
        current_patch = np.multiply(current_patch, mask3d)
        if noise_mult > 0:
            current_patch *= noise_mult
        
        if final_image is None:
            final_image = current_patch
        else:
            final_image = np.clip(np.add(final_image, current_patch), 0, 255)

    
    #We can now return the final np patch and az_util
    return final_image, az_util, return_masks

#loop through
input_fps = 30 #Set according to input speed
attack_runtime = 5 #Maximum time, in seconds, we want our example to be.  Used to limit the time digital attacks take, set to <= 0 to disable

attempts_per_frame = 2 # How many noise patterns should we generate for each frame, 


fourcc = cv2.VideoWriter_fourcc(*'mp4v')

if digital_mode:
    print("DIGITAL MODE ACTIVE")
    print("Out folder: " + current_out_dir.split("\\")[-1])
    vid_idx = 0
    for vidpath in glob.iglob(video_directory + '\\*'):
        print("VIDEO " + str(vid_idx) + ": " + vidpath.split("\\")[-1].split(".")[0])
        az_util_arr = []
        max_perturb = -1
        cap = cv2.VideoCapture(vidpath)
        if not cap.isOpened():
            print("Error on video capture")
            continue
        img_idx = 0
        img_idx_2 = 0

        run_in_progress = True
        first_patch = None
        while run_in_progress:
            ret, frame = cap.read()
            if not ret:
                print("Stream ended.")
                break
            #create temp folder for images
            if run_dictionary:
                print("Dictionary  attempt")
                temp_return_mask = None
                for i in range(0, attempts_per_frame):
                    if temp_return_mask is None:
                        patch, azutil, temp_return_mask = image_to_noise(frame, frame.shape)
                    else:
                        patch, azutil, temp_return_mask = image_to_noise(frame, frame.shape, temp_return_mask, azutil)
                    #Append azutil
                    az_util_arr.append(str(azutil))
                    #Append max noise
                    max_noise_idx = np.unravel_index(patch.argmax(), patch.shape)
                    if patch[max_noise_idx[0]][max_noise_idx[1]][max_noise_idx[2]] > max_perturb:
                        max_perturb = patch[max_noise_idx[0]][max_noise_idx[1]][max_noise_idx[2]]

                    #Add the patch to the frame and save
                    added_im = frame + patch
                    if not cv2.imwrite(os.path.join(current_out_dir + '\\dict', '%d.jpg' % img_idx), added_im):
                        print("COULD NOT WRITE ADDED")
                        print(current_out_dir)
                    img_idx += 1

                #Prevent from going too long
                if attack_runtime > 0 and img_idx >= (input_fps * attempts_per_frame) * attack_runtime:
                    run_in_progress = False
                del temp_return_mask
            
            if run_rt_generation:
                print("Realtime attempt")
                segment_result = None
                for i in range(0, attempts_per_frame):
                    if first_patch is None:
                        victim_imgs = []
                        #Save frame to be loaded
                        if not cv2.imwrite(os.path.join(current_out_dir + '\\rw', 'first_frame.jpg'), frame):
                            print("COULD NOT WRITE ADDED")
                            print(current_out_dir)

                        victim_imgs.append(np.reshape(np.asarray(Image.open(os.path.join(current_out_dir + '\\rw', 'first_frame.jpg')).resize((640, 640))), (3, 640, 640)))
#                        victim_imgs.append(np.reshape(np.asarray(cv2.resize(frame, (640, 640))), (3, 640, 640)))
                        if not(len(victim_imgs) < 1 or victim_imgs[0] is None):
                            run_attack(60, 0.005, victim_imgs, current_out_dir + "\\rw\\PATCH")
                        else:
                            #big oof
                            continue
                            
                        victim_imgs.clear()
                        os.remove(os.path.join(current_out_dir + '\\rw', 'first_frame.jpg'))

                        #Obtain the patch
                        patch_choice = pick_highest_iter(glob.glob(current_out_dir + "\\rw\\PATCH\\save_patch\\*"))
                        print("PATCH CHOICE:")
                        print(patch_choice)
                        first_patch = np.asarray(Image.open(patch_choice))

                    #Process the patch
                    if segment_result is None:
                        segment_result = segment.segment_full_im(segment_image, frame)
   
                    return_mask = segment_result
                    current_patch = np.copy(first_patch)
                    rdim = (frame.shape[1], frame.shape[0])
                    current_patch = cv2.resize(current_patch, rdim, interpolation=cv2.INTER_CUBIC)
                    mask3d = np.stack((return_mask,)*3, axis=-1)

                    current_patch = np.multiply(current_patch, mask3d)

                    added_im_2 = frame + current_patch


                    #Add the patch to the image
                    if not cv2.imwrite(os.path.join(current_out_dir + '\\rw', '%d.jpg' % img_idx_2), added_im_2):
                        print("COULD NOT WRITE ADDED")
                        print(current_out_dir)
                        
                    del current_patch
                    del mask3d
                    
                    img_idx_2 += 1
                    if not run_dictionary:
                        img_idx += 1
                if not run_dictionary and attack_runtime > 0 and img_idx >= (input_fps * attempts_per_frame) * attack_runtime:
                    run_in_progress = False
                del segment_result


                
        
        #Process the images into a video and save the relevant data
        if run_dictionary:
            images = natsorted([img for img in os.listdir(current_out_dir + '\\dict') if img.endswith(".jpg")])
            vidframe = cv2.imread(os.path.join(current_out_dir + '\\dict', images[0]))
            height, width, layers = vidframe.shape

            video = cv2.VideoWriter(current_out_dir + '\\' + vidpath.split("\\")[-1].split(".")[0] + '_dict.mp4', fourcc, input_fps * attempts_per_frame, (width,height))

            for vid_img in images:
                video.write(cv2.imread(os.path.join(current_out_dir + '\\dict', vid_img)))
            video.release()

            #Clean up the image files
            for fname in glob.iglob(current_out_dir + '\\dict\\*.jpg'):
                os.remove(fname)
            #Dump the other stats
            azutil_file = open(current_out_dir + '\\azutil_' + vidpath.split("\\")[-1].split(".")[0] + '.txt', 'w')
            azutil_file.writelines(line + '\n' for line in az_util_arr)
            azutil_file.close()
            az_util_arr.clear()

            max_perturb_file = open(current_out_dir + '\\max_perturb_' + vidpath.split("\\")[-1].split(".")[0] + '.txt', 'w')
            max_perturb_file.write(str(max_perturb))
            max_perturb_file.close()
            
        if run_rt_generation:
            images = natsorted([img for img in os.listdir(current_out_dir + '\\rw') if img.endswith(".jpg")])
            vidframe = cv2.imread(os.path.join(current_out_dir + '\\rw', images[0]))
            height, width, layers = vidframe.shape

            video = cv2.VideoWriter(current_out_dir + '\\' + vidpath.split("\\")[-1].split(".")[0] + '_rw.mp4', fourcc, input_fps * attempts_per_frame, (width,height))

            for vid_img in images:
                video.write(cv2.imread(os.path.join(current_out_dir + '\\rw', vid_img)))
            video.release()

            #Clean up the image files
            for fname in glob.iglob(current_out_dir + '\\rw\\*.jpg'):
                os.remove(fname)
            #Clean up the patch files
            for fname in glob.iglob(current_out_dir + "\\rw\\PATCH\\save_patch\\*"):
                os.remove(fname)


        vid_idx += 1
        
        
            
