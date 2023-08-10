import numpy as np
import torch
import torch.utils.checkpoint
import matplotlib.pyplot as plt
import cv2
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import helper_Functions as hf
import os


def automatic_masking(dir_path, destination_path):
    checkpoint_path = r"/home/shike/AI/SAM/segment-anything/sam_vit_h_4b8939.pth"
    device = "cuda"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamAutomaticMaskGenerator(sam)

    files=sorted(os.listdir(dir_path))

    for file in files:

        img_path = None;
        img_path= dir_path+"/"+file

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = predictor.generate(image=image)

        data_path = destination_path + file + ".npy"
        file = open(data_path, "wb")
        hf.save_mask(masks=masks, file=file)
        file.close()

automatic_masking(r"/home/shike/AI/Data/archive/images/images", r"/home/shike/AI/Data/archive/automatic-masks " )