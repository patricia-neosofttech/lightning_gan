import os
import numpy as np
from PIL import Image
from numpy import savez_compressed

import config


def img_npz():
    # path to the image directory
    dir_data = config.DATA_DIR

    # setting image shape to 32x32
    img_shape = (config.IMG_WIDTH, config.IMG_HEIGHT, 3)

    # listing out all file names
    nm_imgs = np.sort(os.listdir(dir_data))
    print("custom categories:", nm_imgs)

    X_train = []
    for file in nm_imgs:
        try:
            img = Image.open(dir_data + '/' + file)
            img = img.convert('RGB')
            img = img.resize((config.IMG_WIDTH, config.IMG_HEIGHT))
            img = np.asarray(img) / 255
            X_train.append(img)
        except:
            print("something went wrong")

    X_train = np.array(X_train)
    print("Training image shape : ", X_train.shape)

    # save to npy file
    savez_compressed(config.NPZ_PTH, X_train)


img_npz()
