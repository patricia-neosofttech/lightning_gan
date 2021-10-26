import torch
import numpy as np
import matplotlib.pyplot as plt

# load dict of arrays
imgs  = np.load('hatchback_images_32x32.npz')

# extract the first array
data = imgs['arr_0']

# print the array
#print(data)

#check if gpu support available or not
dev = 'cuda:0' if torch.cuda.is_available() == True else 'cpu'
device = torch.device(dev)

#defining helper function
# plot images in a nxn grid

def plot_images(imgs, grid_size=5):
    """
    imgs: vector containing all the numpy images
    grid_size: 2x2 or 5x5 grid containing images
    """

    fig = plt.figure(figsize=(8, 8))
    columns = rows = grid_size
    #plt.title("Training Images")

    for i in range(1, columns * rows + 1):
        plt.axis("off")
        fig.add_subplot(rows, columns, i)
        plt.imshow(imgs[i])
    #plt.show()
    plt.savefig('plots/training_32x32_new.png')
    #plt.savefig('plots/train_op.png')

# pls ignore the poor quality of the images as we are working with 256x256 sized images.
plot_images(imgs['arr_0'], 3)