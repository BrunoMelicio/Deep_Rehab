import argparse
from matplotlib import pyplot as plt
from keras.models import load_model
from training.model import get_personlab
from scipy.ndimage.filters import gaussian_filter
import cv2
import numpy as np
from time import time
from training.config import config
import random
from training.post_proc import *
from training.plot import *

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

def overlay(img, over, alpha=0.5):
    out = img.copy()
    if img.max() > 1.:
        out = out / 255.
    out *= 1-alpha
    if len(over.shape)==2:
        out += alpha*over[:,:,np.newaxis]
    else:
        out += alpha*over    
    return out


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--image', help='An input image to use', default='test.jpg')

args = parser.parse_args()

file_name = args.image

tic = time()

model = get_personlab(train=False, with_preprocess_lambda=True,
                      output_stride=8)

model.load_weights('models/deeprehab_101.h5')

print 'Loading time: {}'.format(time()-tic)

img = cv2.imread(file_name)
img = cv2.resize(img, (0,0), fx=.9, fy=.9)
img = pad_img(img)
print 'Image shape: {}'.format(img.shape)

outputs = model.predict(img[np.newaxis,...])

# Remove batch axes and remove intermediate predictions
outputs = [o[0] for o in outputs]

plt.figure()
img = img[:,:,[2,1,0]]
plt.imshow(img)
n = file_name
plt.axis('off')
plt.savefig("demo_results/input_"+str(n)+".png", dpi=200)


# Here is the output map for right knee
Rknee_map = outputs[0][:,:,config.KEYPOINTS.index('Rknee')]
plt.figure()
plt.imshow(overlay(img, Rknee_map, alpha=0.7))
plt.axis('off')
plt.savefig("demo_results/overlay_"+str(n)+".png", dpi=200)


H = compute_heatmaps(kp_maps=outputs[0], short_offsets=outputs[1])
# Gaussian filtering helps when there are multiple local maxima for the same keypoint.
for i in range(17):
    H[:,:,i] = gaussian_filter(H[:,:,i], sigma=2)
    
plt.figure()
plt.imshow(H[:,:,config.KEYPOINTS.index('Rknee')])
plt.axis('off')
plt.savefig("demo_results/heatmap_"+str(n)+".png", dpi=200)


visualize_short_offsets(offsets=outputs[1], heatmaps=H, keypoint_id='Rknee', img=img, every=8, n=n)


visualize_mid_offsets(offsets= outputs[2], heatmaps=H, from_kp='Rknee', to_kp='Rankle', img=img, every=8,n=n)

pred_kp = get_keypoints(H)
byId = [[] for i in range(23)]

for kp in pred_kp:
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

plt.figure()
plt.imshow(img)

for kp in selected:
    x,y = kp['xy'][0],kp['xy'][1]
    confidence = kp['conf']
    ide = kp['id']
    
    plt.plot(x,y,marker='o',markersize=4)
    #print(x,y)

plt.axis('off')
plt.savefig("demo_results/keypoints_"+str(n)+".png", dpi=200)

print("Results saved in folder /src/demo_results/ !")