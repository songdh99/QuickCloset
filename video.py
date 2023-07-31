import cv2
import os

filepath = '/home/rilab/workspace/sample1.mp4'
video = cv2.VideoCapture(filepath)

if not video.isOpened():
    print("Could not Open :", filepath)
    exit(0)

length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

print("length :", length)
print("width :", width)
print("height :", height)
print("fps :", fps)

try:
    if not os.path.exists(filepath[:-4]):
        os.makedirs(filepath[:-4])
except OSError:
    print ('Error: Creating directory. ' +  filepath[:-4])

count = 0

while(video.isOpened()):
    ret, image = video.read()
    if(int(video.get(1)) % fps == 0): #앞서 불러온 fps 값을 사용하여 1초마다 추출
        cv2.imwrite(filepath[:-4] +"/frame%d.jpg" % count, image)
        print('Saved frame number :', str(int(video.get(1))))
        count += 1
        
video.release()