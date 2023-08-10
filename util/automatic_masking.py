import numpy as np
import torch
import torch.utils.checkpoint
import matplotlib.pyplot as plt
import cv2
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import helper_Functions as hf
import os

checkpoint_path = r"C:\AI\CNIC\SAM\sam_vit_h_4b8939.pth"
device = "cpu"
model_type="vit_h"



sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to(device=device)
predictor = SamAutomaticMaskGenerator(sam)

image_path=r"C:\AI\CNIC\SAM\Data\Image\TestFolder2\test7.png"
image=cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

masks = predictor.generate(image)
plt.imshow(image)
hf.show_anns(masks)
#hf.save_mask(masks, r"C:\AI\CNIC\SAM\Data\Image\TestFolder1\test-musk.npy")
plt.axis('on')
plt.show()
