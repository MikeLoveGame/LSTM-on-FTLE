import numpy as np
import torch
import torch.utils.checkpoint
import matplotlib.pyplot as plt
import cv2
from segment_anything import SamPredictor, sam_model_registry


def show_mask(mask, ax):
    color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.particle_shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def save_mask(masks, file):
    np.save(file, masks)


def load_mask(file):
    datas = np.load(file)
    return datas

def show_anns(anns):
    if (len(anns)==0):
        return
    sorted_anns= sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax =plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []

    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.particle_shape[0], m.particle_shape[1], 3))
        color_mask = np.random.random((3, 7)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

def image_decode(dig, image_code):
    x=str(image_code)
    y=""
    length= len(x)+len(y)
    while(length<dig):
        y+=str('0')
        length=len(x)+len(y)

    return y+x

def image_decode(dig, image_code):
    x=str(image_code)
    y=""
    length= len(x)+len(y)
    while(length<dig):
        y+=str('0')
        length=len(x)+len(y)

    return y+x