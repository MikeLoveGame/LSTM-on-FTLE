import numpy as np
import torch
import torch.utils.checkpoint
import matplotlib.pyplot as plt
import cv2
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import helper_Functions as hf

def image_decode(dig, image_code):
    x=str(image_code)
    y=""
    length= len(x)+len(y)
    while(length<dig):
        y+=str('0')
        length=len(x)+len(y)

    return y+x



checkpoint_path= r"C:\AI\CNIC\SAM\sam_vit_h_4b8939.pth"
device = "cpu"
model_type="vit_h"

sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to(device=device)

'''predictor = SamAutomaticMaskGenerator(sam)

image_code=0;
image_dirpath= r"C:\AI\CNIC\SAM\Data\Image\pstupiansucaiku\pstupiansucaiku"


for i in range(1, 3):
    img=image_decode(4, i)
    image_path=image_dirpath+"\\"+str(img)+".jpg"
    image=cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = predictor.generate(image)
    file_path=r"C:\AI\CNIC\SAM\Data\Mask"+img
    file = open(file_path, "wb")
    hf.save_mask(masks=masks, file=file)
    file.close()
'''

predictor = SamPredictor(sam)

image_code=0;
image_dirpath= r"C:\AI\CNIC\SAM\Data\Image\pstupiansucaiku\pstupiansucaiku"


for i in range(1, 3):
    img=image_decode(4, i)
    image_path=image_dirpath+"\\"+str(img)+".jpg"
    image=cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    input_point = np.array([[600, 750], [300,300]])
    input_label = np.array([1, 1])
    masks, scores, logits= predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output= True)
    file_path=r"C:\AI\CNIC\SAM\Data\Mask"+img
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))

        plt.imshow(image)
        hf.show_points(input_point, input_label, plt.gca())
        hf.show_mask(mask, plt.gca())
        plt.title(f"Mask {i + 1}, Score : {score:.3f}", fontsize=18)
        plt.axis('on')
        plt.show()

    file = open(file_path, "wb")
    hf.save_mask(masks=masks, file=file)
    file.close()

#hf.save_mask(masks, "C:\AI\CNIC\SAM\Data\Mask\\testData.npy")

plt.figure(figsize=(10,10))
plt.imshow(image)
hf.show_anns(masks)
plt.axis('off')
plt.show()






