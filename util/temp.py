import numpy as np
import torch
import torch.utils.checkpoint
import matplotlib.pyplot as plt
import cv2
from segment_anything import SamPredictor, sam_model_registry

from typing import Optional, Tuple

def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.particle_shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    
def save_mask(masks : np.ndarray, file):
    np.save(file, masks)
    
def load_mask(file) :
    datas=np.load(file)
    return datas


checkpoint_path= r"C:\AI\CNIC\SAM\sam_vit_h_4b8939.pth"
device = "cpu"
model_type="vit_h"



sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to(device=device)

predictor = SamPredictor(sam)

path= r"C:\AI\CNIC\SAM\Data\Image\TestFolder1\img2.jpg"
image = cv2.imread(path)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor.set_image(image)

input_point = np.array([[700, 1000]])
input_label = np.array([1])

masks, scores, logits=predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output= True)

for i , (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    
    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    show_mask(mask, plt.gca())
    plt.title(f"Mask {i+1}, Score : {score:.3f}", fontsize=18)
    plt.axis('on')
    plt.show()

'''file = open (r"/Data/tempData.npy", str('wb'))

save_mask(masks, file)

file.close()

file2 = open(r"/Data/tempData.npy", str('rb'))

masks = load_mask(file2)
file2.close()

for i, (mask) in enumerate(zip(masks)):
    mask_np = np.asarray(mask, bool)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(mask_np, plt.gca())
    plt.axis('off')
    plt.show()'''
    
    





