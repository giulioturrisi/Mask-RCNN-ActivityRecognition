# coding=utf-8
#TO RUN
#python3 -i testVideo.py image --source /media/giulio/TOSHIBA\ EXT/dataset_Turrisi/test/normalImages/   --model /home/giulio/Scrivania/mask_rcnn_visiope_0080.h5 
#
import random
import os
import sys
import random
import colorsys
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import time
import ntpath
#
from collections import Counter
#
from PIL import Image
import glob

# Root directory of the project
ROOT_DIR = os.path.abspath("./Mask_RCNN")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
# Import COCO config
import json
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def colors(n):

    ret = []
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for i in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        ret.append((r,g,b))
    random.shuffle(ret)
    return ret

def fill_mask(img,mask,channel,color,h,w):
    for r in range(h):
        for c in range(w):
            if mask[r][c][channel]:
                img[r][c][0] = color[0]
                img[r][c][1] = color[1]
                img[r][c][2] = color[2]
    return img


EVALUATION_CLASSES = []

'''path = "/home/edoardo/Desktop/Turrisi_Lerario/dataset_Turrisi"
jsonName = "train.json"
jsonPath = path + "/" + jsonName
nomeBase = "image"
pathMask = path + "/pngImages/"
b = json.load(open(jsonPath))
classes = []
classes.append("BG")
image_ids = []
EVALUATION_CLASSES.append('BG')
for xx in range(len(b)):
    name = xx
    if b[xx]['Label'] == "Skip":
        continue
    else:
        image_ids.append(xx)
        for x in b[xx]['Label'].keys():
            name = x
            if name not in classes:
                classes.append(name)
                EVALUATION_CLASSES.append(name)
print(classes)
#txtPath = "/home/giulio/Scrivania/Università/Vision_and_Perception/Object.txt"
#textfile = open(txtPath, "r")
class_ids = []
coco_classes = ['person','car','traffic light','cat','dog','horse','cow','backpack','frisbee','suitcase','sports ball','couch','scissors','toothbrush','hair drier','handbag','bench','sink','bowl','stop sign']
#if not class_ids:
#    # All classes
#    for cl in coco_classes:
#        class_ids.extend(coco.getCatIds(catNms=[cl]))



for CLASS in coco_classes:
    EVALUATION_CLASSES.append(CLASS)
print(EVALUATION_CLASSES)
'''
COLORS = [(231, 119, 39), (45, 189, 109), (115, 3, 179), (31, 175, 95), (52, 196, 116), (217, 105, 25), (38, 182, 102), (238, 126, 46), (185, 73, 249), (59, 203, 123), (129, 17, 193), (171, 59, 235), (108, 252, 172), (10, 154, 74), (80, 224, 144), (94, 238, 158), (66, 210, 130), (150, 38, 214), (224, 112, 32), (178, 66, 242), (17, 161, 81), (157, 45, 221), (210, 98, 18), (87, 231, 151), (122, 10, 186), (24, 168, 88), (143, 31, 207), (101, 245, 165), (3, 147, 67), (252, 140, 60), (164, 52, 228), (73, 217, 137), (136, 24, 200), (245, 133, 53), (142,213,11), (74,36,99), (100,22,200), (90,37,67), (125,55,252), (65,71,121), (171,65,29), (9,45,192), (158,59,147), (112,118,82), (0,23,13), (32,42,72), (221,248,243), (180,220,2)]

classes_color = colors(43)

class InferenceConfig(Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    NUM_CLASSES = 22+1+20
    IMAGE_MAX_DIM = 768

    # Give the configuration a recognizable name
    NAME = "vision"


def returnName(id):
    if(id == 1):
        return 'Lawn'
    if(id == 2):
        #return 'Camel'
        return 'Rope'
    if(id == 3):
        #return 'Rope'
        return 'Camel'
    if(id == 4):
        return 'Sky'
    if(id == 5):
        return 'Cowboy Hat'
    if(id == 6):
        return 'Hand'
    if(id == 7):
        return 'nail_clipper'
    if(id == 8):
        return 'scissor'
    if(id == 9):
        return 'Horse saddle'
    if(id == 10):
        return 'Horse bridle'
    if(id == 11):
        return 'Bull'
    if(id == 12):
        return 'Spear'
    if(id == 13):
        return 'Calf'
    if(id == 14):
        return 'Brush'
    if(id == 15):
        return 'Shower_handle'
    if(id == 16):
        return 'Water'
    if(id == 17):
        return 'Red fabric'
    if(id == 18):
        return 'Riding Hat'
    if(id == 19):
        return 'Polo stick'
    if(id == 20):
        return 'Shaver'
    if(id == 21):
        return 'horseshoe'
    if(id == 22):
        return 'Dog Leash'
    if(id == (23)):
        return 'Person'
    if(id == (24)):
        return 'car'
    if(id == (25)):
        return 'traffic light'
    if(id == (26)):
        return 'cat'
    if(id == (27)):
        return 'dog'
    if(id == (28)):
        return 'horse'
    if(id == (29)):
        return 'cow'
    if(id == (30)):
        return 'backpack'
    if(id == (31)):
        return 'frisbee'
    if(id == (32)):
        return 'suitcase'
    if(id == (33)):
        return 'sports ball'
    if(id == (34)):
        return 'couch'
    if(id == (35)):
        return 'scissor'
    if(id == (36)):
        return 'toothbrush'
    if(id == (37)):
        return 'hair_drier'
    if(id == (38)):
        return 'handbag'
    if(id == (39)):
        return 'bench'
    return "NO"

#new_json = {}
def create_datasetVideo(model,pathVideo):
    #random_colors(len(CLASSES))
    print(pathVideo)
    ###########
    #new_json[pathVideo] = {'activity' : activity_video}
    #new_json[pathVideo] = {'masks' : []}
    
    cap = cv2.VideoCapture(pathVideo)

    if cap:
        fps = cap.get(cv2.CAP_PROP_FPS)
        framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        most_probable = []
        temp_probable = []
        
        for f in range(int(framecount)):
            ret, frame = cap.read()
            if(f % 50 == 0):
                #ret, frame = cap.read()
                #print(frame)
                if not ret:
                    break
                #print("Frame {} of {}".format(f,framecount))
                results = model.detect([frame])
                r = results[0]
                #print("IL RISULTATO e'")
                #print(results)
                #quasi = 0
                for i,id_class in enumerate(r['class_ids']):
                    #if(quasi == 7):
                    #    break
                    true_call_name = returnName(id_class)
                    #########
                    if(true_call_name != "Sky"):
                        temp_probable.append(true_call_name)
                    ############
                    #cv2.putText(frame, true_call_name +": "+str(r['scores'][i]),(y,x),1,2,COLORS[id_class],6)
            if f % 400 == 0:
                #for scorro in temp_probable:
                print("Frame {} of {}".format(f,framecount))
                count = Counter(temp_probable)
                print(count)
                print("I piu probabili sono")
                print(count.most_common(6))
                common = count.most_common(6)
                bo = []
                if(len(common) > 0):
                    if(common[0]):
                        print(common[0][0])
                        bo.append(common[0][0])
                        if(common[0][1] > 4):
                            bo.append(common[0][0])
                if(len(common) > 1):
                    if(common[1]):
                        print(common[1][0])
                        bo.append(common[1][0])
                        if(common[1][1] > 4):
                            bo.append(common[1][0])
                if(len(common) > 2):
                    if(common[2]):
                        print(common[2][0])
                        bo.append(common[2][0])
                        if(common[2][1] > 4):
                            bo.append(common[2][0])                        
                if(len(common) > 3):
                    if(common[3]):
                        print(common[3][0])
                        bo.append(common[3][0])
                        if(common[3][1] > 4):
                            bo.append(common[3][0])
                if(len(common) > 4):
                    if(common[4]):
                        print(common[4][0])
                        bo.append(common[4][0])
                        if(common[4][1] > 4):
                            bo.append(common[4][0])
                if(len(common) > 5):
                    if(common[5]):
                        print(common[5][0])
                        bo.append(common[5][0])
                        if(common[5][1] > 4):
                            bo.append(common[5][0])
                if(len(common) > 6):
                    #common[6][1] > 1
                    if(common[6]):
                        print(common[6][0])
                        bo.append(common[6][0])
                        if(common[6][1] > 4):
                            bo.append(common[6][0])


                if(len(common) > 0):
                    most_probable.append(bo)
                print("#################")

                ##############
                temp_probable = []
        cap.release()
        print("IN THE ENDDD")
        print(most_probable)
        return most_probable

def testVideo(model,pathVideo):
    #random_colors(len(CLASSES))
    print(pathVideo)
    cap = cv2.VideoCapture(pathVideo)
    #create dataset folder
    if not os.path.exists('resultsVideo/'):
        os.makedirs('resultsVideo/')

    if cap:
        fps = cap.get(cv2.CAP_PROP_FPS)
        framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('resultsVideo/'+str(ntpath.basename(pathVideo)),fourcc, fps, (width,height))
        for f in range(int(framecount)):
            ret, frame = cap.read()
            if not ret:
                break
            if f % 100 == 0:
                print("Frame {} of {}".format(f,framecount))
            results = model.detect([frame])
            r = results[0]
            #print("IL RISULTATO e'")
            #print(results)
            overlay = frame.copy()
            for i,id_class in enumerate(r['class_ids']):
                x = r['rois'][i][0]
                y = r['rois'][i][1]
                w = r['rois'][i][2]
                h = r['rois'][i][3]
                img = fill_mask(overlay,r['masks'],i,COLORS[id_class],height,width)
                opacity = 0.4
                cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
                cv2.rectangle(frame,(y,x),(h,w),COLORS[id_class],2)
                true_call_name = returnName(id_class)
                cv2.putText(frame, true_call_name +": "+str(r['scores'][i]),(y,x),1,2,COLORS[id_class],6)
            out.write(frame)
        cap.release()
        out.release()

def testImage(model,pathImage):
    #random_colors(len(CLASSES))
    #print(pathVideo)
    #cap = cv2.VideoCapture(pathVideo)
    #create dataset folder
    if not os.path.exists('resultsImage/'):
        os.makedirs('resultsImage/')

    # read image
    for filename in os.listdir(pathImage):
        try:
            img = cv2.imread(pathImage + '/' + filename, cv2.IMREAD_UNCHANGED)
            # get dimensions of image
            dimensions = img.shape 
            # height, width, number of channels in image
            height = img.shape[0]
            width = img.shape[1]
            channels = img.shape[2]
            results = model.detect([img])
            print(results)
            r = results[0]
            overlay = img.copy()
            for i,id_class in enumerate(r['class_ids']):
                x = r['rois'][i][0]
                y = r['rois'][i][1]
                w = r['rois'][i][2]
                h = r['rois'][i][3]
                img_mask = fill_mask(overlay,r['masks'],i,COLORS[id_class],height,width)
                opacity = 0.4
                cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
                cv2.rectangle(img,(y,x),(h,w),COLORS[id_class],2)
                true_call_name = returnName(id_class)
                cv2.putText(img, true_call_name +": "+str(r['scores'][i]),(y,x),1,2,(106, 0, 0),6)
            #cv2.imshow('image',img)
            #cv2.waitKey(0)
            print('/home/giulio/Scrivania/Università/Vision_and_Perception/resultsImage/' + filename + ".jpg")
            cv2.imwrite('/home/giulio/Scrivania/Università/Vision_and_Perception/resultsImage/' + filename + ".jpg", img)
        except Exception as e:
            continue

if __name__ == '__main__':

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect TeamSportsActivities.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'video' or 'image' ")
    parser.add_argument('--source', required=True,
                        metavar="/path/to/source/file",
                        help='Directory of the Source')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/model.h5",
                        help="Path to weights .h5 file")
    args = parser.parse_args()

    config = InferenceConfig()
    config.display()
    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=DEFAULT_LOGS_DIR, config=config)
    # Load weights trained on MS-COCO
    model.load_weights(args.model, by_name=True)
    print("QUI ARRIVOO?")
     # Train or evaluate
    if(args.command == 'video'):
        #file_names = next(os.walk(args.source))[2]
        #for f in os.listdir(args.source):
        #pathVideo = os.path.join(args.source, f)
        #print(f)
        testVideo(model, args.source)
    elif(args.command == 'create_dataset_Video'):
        new_json = {}
        file_names = next(os.walk(args.source))[1]
        print(file_names)
        for f in file_names:
            print("inizio con")
            print(f)
            complete_path = args.source + "/" + f
            videos = next(os.walk(complete_path))[2]
            print(videos)
            for video in videos:
                pathVideo = os.path.join(complete_path, video)
                #########CREO IL JSON###################
                new_json[pathVideo] = {'activity' : f,'masks' : []}
                ########################################
                #print(new_json)
                ##########CREO DATASET VIDEO##
                new_mask =  create_datasetVideo(model, pathVideo)
                for separate_mask in new_mask:
                    new_json[pathVideo]['masks'].append(separate_mask)
                print(new_json)
        with open('dataset_video_3.txt', 'w') as outfile:
            json.dump(new_json, outfile)  
    else:
        testImage(model,args.source)
