#!/usr/bin/python3

# https://blog.roboflow.ai/how-to-convert-annotations-from-voc-xml-to-coco-json/
import mmcv
import os
import argparse
import shutil
import cv2
import json
import xml.etree.ElementTree as ET
from typing import Dict, List
from tqdm import tqdm
import re
from glob import glob
import hashlib
from detectron2.structures import BoxMode
import sys
sys.path.append("..")
from utils import *


def get_image_info (filename):
    #img_id = os.path.basename(filename).split("_")[0]
    img_id = os.path.basename(filename)
    img = cv2.imread(filename)
    width = img.shape[1]
    height = img.shape[0]

    image_info = {
        'file_name': os.path.abspath(filename),
        'height': height,
        'width': width,
        'id': img_id
    }
    return image_info


# recreate data folder
annDir = "/data/uke/data/knee.lat/asm_detect/train"
shutil.rmtree(annDir, ignore_errors = True)
os.makedirs(annDir)


# also we need the test set here for final evalution
print ("Creating .json for train set")
annList = glob("../annotations/*.csv")
print ("\tFound ", len(annList), "annotations")

output_json_dict = {
    "images": [],
    "type": "instances",
    "annotations": [],
    "categories": []
}

for ann in annList:
    # load csv
    currentFrame = pd.read_csv(ann, header = 0, index_col = 0).T.iloc[0].copy()
    # do we have a annotation?
    if int(currentFrame["ASM_x"]) == 0:
        continue
    if int(currentFrame["Invalid"]) == 1:
        #print ("INV")
        continue

    dcmName = currentFrame["DCM_File"]
    #print ("loading " + dcmName)

    # sitk load image
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([dcmName])
    baseimage = reader.Execute()
    image = getWindowed (baseimage, None, None)
    cBox_x = int(currentFrame["ASM_x"])
    cBox_y = int(currentFrame["ASM_y"])
    cWidth = int(currentFrame["ASM_Width"])
    cHeight = int(currentFrame["ASM_Height"])

    # write it to our stash

    pngFile = os.path.join("/data/uke/data/knee.lat/asm_detect/train/", os.path.basename(currentFrame["series_instance_uid"] + ".png"))
    # same processing as in annotation for now
    image = image[0,:,:]
    image = cv2.resize (image, (image.shape[1]//3, image.shape[0]//3))
    image = np.stack((image,)*3, axis = -1)

    cv2.imwrite(pngFile, image)

    img_info = get_image_info (pngFile)
    img_id = img_info['id']
    output_json_dict['images'].append(img_info)

    ann = {
        'area': cWidth * cHeight,
        'iscrowd': 0,
        'bbox_mode': BoxMode.XYWH_ABS,
        'bbox': [cBox_x, cBox_y, cWidth, cHeight],
        'category_id': 1,
        'ignore': 0,
        'segmentation': []  # This script is not for segmentation
    }

    ann.update({'image_id': img_id, 'id': abs(hash("box" + os.path.basename(pngFile))) % (10 ** 16) })
    output_json_dict['annotations'].append(ann)

category_info = {'supercategory': 'none', 'id': 1, 'name': "ASM"}
output_json_dict['categories'].append(category_info)
output_jsonpath= "/data/uke/data/knee.lat/asm_detect/train.json"
mmcv.dump(output_json_dict, output_jsonpath)


#
