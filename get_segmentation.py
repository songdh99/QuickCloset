import cv2
from PIL import Image
import pickle
import json
import numpy as np

img = Image.open('./data/resized_segmentation_img.png')
img_w ,img_h = img.size

img = np.array(img)
color_img=np.zeros((img_h,img_w,3), dtype=np.uint8)

for y_idx in range(img.shape[0]):
    for x_idx in range(img.shape[1]):    
        tmp = img[y_idx][x_idx]
        if np.array_equal(tmp, [0,0,0]):
            color_img[y_idx][x_idx] = [0, 0, 0]
        elif np.array_equal(tmp, [255,0,0]):
            color_img[y_idx][x_idx] = [0, 0, 255] #머리카락
        elif np.array_equal(tmp, [0,0,255]):
            color_img[y_idx][x_idx] = [255, 0, 0] #머리
        elif np.array_equal(tmp, [85, 51, 0]):
            color_img[y_idx][x_idx] = [0, 255, 0] #목
        elif np.array_equal(tmp, [255, 85, 0]):
            color_img[y_idx][x_idx] = [0, 85, 255] #몸통
        elif np.array_equal(tmp, [0, 255, 255]):
            color_img[y_idx][x_idx] = [255, 255, 0] #왼팔
        elif np.array_equal(tmp, [51, 170, 221]):
            color_img[y_idx][x_idx] = [221, 170, 51] #오른팔
        elif np.array_equal(tmp, [0, 85, 85]):
            color_img[y_idx][x_idx] = [85, 0, 85] #바지
        elif np.array_equal(tmp, [0, 0, 85]):
            color_img[y_idx][x_idx] = [85, 85, 0] #원피스
        elif np.array_equal(tmp, [0, 128, 0]):
            color_img[y_idx][x_idx] = [0, 128, 128] #치마
        elif np.array_equal(tmp, [177, 255, 85]):
            color_img[y_idx][x_idx] = [85, 255, 177] #왼다리
        elif np.array_equal(tmp, [85, 255, 170]):
            color_img[y_idx][x_idx] = [170, 255, 85] #오른다리
        elif np.array_equal(tmp, [0, 119, 221]):
            color_img[y_idx][x_idx] = [221, 119, 0] #외투
        else:
            color_img[y_idx][x_idx] = [0, 0, 0]
            
img=cv2.resize(color_img,(768,1024),interpolation=cv2.INTER_NEAREST)
bg_img = Image.fromarray(img)
bg_img.save("./HR-VITON-main/test/test/image-parse-v3/00001_00.png")
