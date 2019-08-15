# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import sys
import glob
import image_processor
sys.path.append(os.path.join(os.getcwd(), 'mask_rcnn'))
from mask_rcnn.mrcnn_model import MaskRcnnModel
from tqdm import tqdm


OUT_DIR = 'results'

# サーバー起動時にモデルインスタンス作成
mrcnn_model = MaskRcnnModel()


def gray(img_path):
    ###
    #keep human area color and make other area gray
    ###
    img = cv2.imread(img_path)
    result = mrcnn_model.detect(img)
    res_img = image_processor.make_image_gray(result, img)
    name = os.path.basename(img_path)
    name = os.path.splitext(name)[0]
    out_path = os.path.join(OUT_DIR, '{}_gray.jpg'.format(name))
    cv2.imwrite(out_path, res_img)

def blur(img_path):
    ###
    #keep human area focused and make other area blur
    ###
    img = cv2.imread(img_path)
    result = mrcnn_model.detect(img)
    res_img = image_processor.make_image_blur(result, img)
    name = os.path.basename(img_path)
    name = os.path.splitext(name)[0]
    out_path = os.path.join(OUT_DIR, '{}_blur.jpg'.format(name))
    cv2.imwrite(out_path, res_img)

if __name__ == '__main__':
    input_dir = '/Users/konotatsuya/images4blur'
    imgs = glob.glob(os.path.join(input_dir, '*.jpg'))
    for img in tqdm(imgs):
        blur(img)