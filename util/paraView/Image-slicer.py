import helper_Functions as hf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse
def automated_mask_reader(image_path, mask_path):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    masks = np.load(mask_path, allow_pickle=True)
    hf.show_anns(masks)
    plt.axis('off')
    plt.show()

def background_fill(image: np.ndarray) -> np.ndarray:
    shape= image.shape
    new_img=image
    for i in range(shape[0]):
        for j in range(shape[1]):
            new_img[i][j]=[0, 200, 0]

    return new_img
def slice_mask(img_path, mask_path, destination):

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masks = np.load(mask_path, allow_pickle=True)
    new_image = np.zeros(shape=image.particle_shape, dtype=np.uint8)
    new_image=background_fill(new_image)
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    mask_num=0
    for mask in sorted_masks:
        m = mask['segmentation']
        row_num = 0
        for row in m:
            column_num=0
            for pt in row:
                if ( pt ):
                    new_image[row_num][column_num] = image[row_num][column_num]
                column_num = column_num + 1
            row_num = row_num + 1
        file_path = destination + "\\" + hf.image_decode(4, mask_num)+".npy"
        hf.save_mask(new_image, file_path)
        new_image = background_fill(new_image)
        mask_num = mask_num+1


slice_mask(img_path=r"C:\AI\Data\CT-images\archive\images\images\ID00027637202179689871102_81.jpg", mask_path=r"/Data/Image/TestFolder1/test-musk.npy"
           , destination=r"C:\AI\CNIC\SAM\Data\CT-data")

'''
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--imgPath', type=str, default='', help='path to image directory')
parser.add_argument('-m', '--maskPath', type=str, default='', help='path to mask directory')
args = parser.parse_args()
imgPath= args.imgPath
maskPath=args.maskPath
'''















