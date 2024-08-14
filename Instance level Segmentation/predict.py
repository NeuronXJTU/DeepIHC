#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from yolact import YOLACT

import os

from tqdm import tqdm

if __name__ == "__main__":
    yolact = YOLACT()
    mode = "dir_predict"
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"
    img_names = os.listdir(dir_origin_path)
    for img_name in tqdm(img_names):
        if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            image_path  = os.path.join(dir_origin_path, img_name)
            image       = Image.open(image_path)
            r_image   = yolact.detect_image(image)
            if not os.path.exists(dir_save_path):
                os.makedirs(dir_save_path)
            r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)


