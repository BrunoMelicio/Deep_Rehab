from matplotlib import pyplot as plt

from keras.models import load_model
from training.model import get_personlab
from scipy.ndimage.filters import gaussian_filter

import numpy as np
from time import time
from training.config import config
import random
from training.post_proc import *

import cv2

KEYPOINTS = [
        "nose",         # 0
        # "neck",       
        "Rshoulder",    # 1
        "Relbow",       # 2
        "Rwrist",       # 3
        "Lshoulder",    # 4
        "Lelbow",       # 5
        "Lwrist",       # 6
        "Rhip",         # 7
        "Rknee",        # 8
        "Rankle",       # 9
        "Lhip",         # 10
        "Lknee",        # 11
        "Lankle",       # 12
        "Reye",         # 13
        "Leye",         # 14
        "Rear",         # 15
        "Lear",         # 16
        "Rheel",        # 17 - extra starts here
        "Rbigtoe",      # 18
        "Rlittletoe",   # 19
        "Lheel",        # 20
        "Lbigtoe",      # 21
        "Llittletoe"    # 22
    ]


EDGES = [
        (0, 14),
        (0, 13),
        (0, 4),
        (0, 1),
        (14, 16),
        (13, 15),
        (4, 10),
        (1, 7),
        (10, 11), #new hips
        (1,  4), #new shoulders
        (7, 10),
        (7, 8),
        (11, 12),
        (8, 9),
        (4, 5),
        (1, 2),
        (5, 6),
        (2, 3),
        (9,20), #extra starts here
        (9,21),
        (9,22),
        (12,17),
        (12,18),
        (12,19)
    ]

# Pad image appropriately (to match relationship to output_stride as in training)
def pad_img(img, mult=16):
    h, w, _ = img.shape
    
    h_pad = 0
    w_pad = 0
    if (h-1)%mult > 0:
        h_pad = mult-((h-1)%mult)
    if (w-1)%mult > 0:
        w_pad = mult-((w-1)%mult)
    return np.pad(img, ((0,h_pad), (0,w_pad), (0,0)), 'constant')

def draw_kps(preds,img):
    #print(preds)
    for kp in preds:
        k = kp['xy']
        ide = kp['id']
        #if ide in [8,11,17,18,19,20,21,22]:
        cv2.circle(img, (k[0],k[1]), 2, (255,0,0), -1)
        
def select_kp(preds):
    byId = [[] for i in range(23)]
    for kp in preds:
        byId[kp['id']].append(kp)


    selected = []
    for i,group in enumerate(byId):
        #print(group)
        if(len(group) < 1):
            continue
        maximum = 0
        ind = 0
        for j,elem in enumerate(group):
            if elem['conf'] > maximum:
                maximum = elem['conf']
                ind = j
        selected.append(group[ind])
        
    return selected

def draw_skeleton(preds, img):
    for (v1,v2) in EDGES:
        kp1 = None
        kp2 = None
        for kp in preds:
            if(kp['id'] == v1):
                kp1 = tuple(list(kp['xy']))
                #print(kp1)
            if(kp['id'] == v2):
                kp2 = tuple(list(kp['xy']))
            if(kp1 != None and kp2 != None):
                break
        if kp1!=None and kp2!=None:
            #print("X:",abs(kp1[0]-kp2[0]))
            #print("Y:",abs(kp1[1]-kp2[1]))
            if (v1 == 1 and v2 == 2) and (abs(kp1[0]-kp2[0]) > 20 or abs(kp1[1]-kp2[1]) > 100):
                continue
            if (v1 == 2 and v2 == 3) and (abs(kp1[0]-kp2[0]) > 100 or abs(kp1[1]-kp2[1]) > 100):
                continue
            #if v1 == 2 and v2 == 3:
            #    cv2.line(img, kp1, kp2, (255,0,0), 2)
            #    print("X:",abs(kp1[0]-kp2[0]))
            #    print("Y:",abs(kp1[1]-kp2[1]))
            cv2.line(img, kp1, kp2, (255,0,0), 2)
        else:
            continue
            
tic = time()
#model = get_personlab(train=False, with_preprocess_lambda=True,
#                      intermediate_supervision=True,
#                      intermediate_layer='res4b12_relu',
#                      build_base_func=get_resnet101_base,
#                      output_stride=16)
model = get_personlab(train=False, with_preprocess_lambda=True,
                      output_stride=8)
print 'Loading time: {}'.format(time()-tic)

model.load_weights('models/deeprehab_101.h5')

cap = cv2.VideoCapture('test_video.mp4')
#cap.open('vid.mp4')
#print(cap.isOpened())

fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
#fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
#fourcc = cv2.VideoWriter_fourcc(*'XVID')

#out = cv2.VideoWriter('overlay2_out.avi', fourcc, 20.0, (433,769))#289,225 897,513 385,225 641,369 577,337
#out = cv2.VideoWriter('output_fp.avi', fourcc, 20.0, (961,545))

font = cv2.FONT_HERSHEY_SIMPLEX

fpss = []
inferences = []

while(cap.isOpened()):
    ret, frame = cap.read()
    
    if ret == False:
        break

    #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
    img = cv2.resize(frame, (0,0), fx=.4, fy=.4) #.4
    img = pad_img(img)
    #print(img.shape)

    start = time()
    
    outputs = model.predict(img[np.newaxis,...])

    inferences.append((time()-start) * 1000)

    outputs = [o[0] for o in outputs]

    H = compute_heatmaps(kp_maps=outputs[0], short_offsets=outputs[1])
    #Gaussian filtering helps when there are multiple local maxima for the same keypoint.
    for i in range(23):
        H[:,:,i] = gaussian_filter(H[:,:,i], sigma=2)
        
    pred_kp = get_keypoints(H)
    
    selected = select_kp(pred_kp)
    
    #draw_kps(pred_kp, img)
    #draw_kps(selected, img)
    
    draw_skeleton(selected, img)
        
    #out.write(img)

    fps  = 1 / (time() - start)
    fpss.append(fps)

    cv2.putText(img, 'FPS: '+ str(round(fps,2)), (10, 15), font, 0.5, 
                 (255,0,0), 1, cv2.LINE_AA)
    
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Average FPS:", sum(fpss)/len(fpss))
print("Average Inference:", sum(inferences)/len(inferences))