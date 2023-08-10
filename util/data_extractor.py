import helper_Functions as hf
import numpy as np
import matplotlib.pyplot as plt
import cv2


def show_point_masks():
    file = open(r"C:\AI\CNIC\SAM\Data\Mask\mask.npy", "rb")
    masks = np.load(file, allow_pickle=True)
    image_dirpath = r"C:\AI\CNIC\SAM\Data\Image\pstupiansucaiku\pstupiansucaiku"
    mask_dirpath = r"C:\AI\CNIC\SAM\ServerData\2\Data\images\maskSet1"
    for i in range(1, 100):

        img = hf.image_decode(4, i)
        image_path = image_dirpath + "\\" + str(img) + ".jpg"
        mask_path = mask_dirpath + "\\" + str(img) + ".npy"
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = hf.load_mask(mask_path)
        for i, mask in enumerate(zip(masks)):
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            input_point = np.array([[300, 350]])
            input_label = np.array([1])
            hf.show_points(input_point, input_label, plt.gca())
            hf.show_mask(masks, plt.gca())
            plt.axis('on')
            plt.show()

def automated_mask_reader():

    image_path = r"C:\AI\Data\CT-images\archive\images\images\ID00027637202179689871102_81.jpg"
    mask_path = r"C:\AI\CNIC\SAM\Data\CT-data"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    masks = np.load(mask_path, allow_pickle=True)
    hf.show_anns(masks)
    plt.axis('off')
    plt.show()
def view_masks():
    imagedir_path = r"C:\AI\CNIC\SAM\Data\CT-data"
    ax = plt.gca()

    for i in range(0, 80):
        img = hf.image_decode(4, i)
        image_path = imagedir_path + "\\" + str(img) + ".npy"
        img = np.load(file=image_path, allow_pickle=True )
        #ax.set_autoscale_on(False)
        plt.imshow(img)
        plt.show()
view_masks()