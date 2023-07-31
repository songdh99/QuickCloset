import re
import os
import numpy as np
os.system(f'ffmpeg -r 30 -i ims/2/a/2a.*.png'
              f' -crf 30 ims/2/a/2a.mp4')

path = "ims/2/a"
paths = [os.path.join(path , i ) for i in os.listdir(path) if re.search(".png$", i )]
## 정렬 작업
store1 = []
store2 = []
for i in paths :
    if len(i) == 19 :
        store2.append(i)
    else :
        store1.append(i)

paths = list(np.sort(store1)) + list(np.sort(store2))
#len('ims/2/a/2a.2710.png')