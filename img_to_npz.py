import os
import numpy as np
from PIL import Image
from numpy import savez_compressed


def img_npz():
    # path to the image directory
    dir_data = "vehicle_ds/images/hatchback"

    # setting image shape to 32x32
    img_shape = (32, 32, 3)

    # listing out all file names
    nm_imgs = np.sort(os.listdir(dir_data))
    print("custom categories:", nm_imgs)

    X_train = []
    for file in nm_imgs:
        try:
            img = Image.open(dir_data + '/' + file)
            img = img.convert('RGB')
            img = img.resize((32, 32))
            img = np.asarray(img) / 255
            X_train.append(img)
        except:
            print("something went wrong")

    X_train = np.array(X_train)
    print("Training image shape : ", X_train.shape)

    # save to npy file
    savez_compressed('hatchback_images_32x32.npz', X_train)


img_npz()
