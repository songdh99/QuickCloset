import os, sys
import cv2
from PIL import Image
import numpy as np
import glob
import warnings
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--background', type=bool, default=True, help='Define removing background or not')
    opt = parser.parse_args()
    
    # print("\nGenerate Densepose image using detectron2 library\n")
    # terminnal_command ="python3 ./detectron2/projects/DensePose/apply_net.py dump ./detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml \
    # https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl \
    # origin.jpg --output output.pkl -v"
    # os.system(terminnal_command)
    # terminnal_command ="python3 get_densepose.py"
    # os.system(terminnal_command)
    
    
    print("\nRun HR-VITON to generate final image\n")
    os.chdir("./HR-VITON-main")
    terminnal_command = "python3 test_generator.py --cuda True --test_name test1 --tocg_checkpoint /home/ri-1080/song_ws/TryYours-Virtual-Try-On/HR-VITON-main/eval_models/weights/v0.1/mtviton.pth --gpu_ids 0 --gen_checkpoint /home/ri-1080/song_ws/TryYours-Virtual-Try-On/HR-VITON-main/eval_models/weights/v0.1/gen.pth --datasetting unpaired --data_list t2.txt --dataroot ./test" 
    os.system(terminnal_command)

    # Add Background or Not
    l=glob.glob("./Output/*.png")


    for i in l:
        img=cv2.imread(i)
        cv2.imwrite(i,img)
    # Add Background
    # if opt.background:
    #     for i in l:
    #         img=cv2.imread(i)
    #         img=cv2.bitwise_and(img,img,mask=mask_img)
    #         img=img+back_ground
    #         cv2.imwrite(i,img)

    # # Remove Background
    # else:
    #     for i in l:
    #         img=cv2.imread(i)
    #         cv2.imwrite(i,img)

    os.chdir("../")
    cv2.imwrite("./static/finalimg2.png", img)