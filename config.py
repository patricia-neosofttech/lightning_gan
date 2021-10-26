# path to the image directory
DATA_DIR = "vehicle_ds/images/hatchback"

# img file path
IMG_PTH = 'plots/train_op.png'

#numpy file path
NPZ_PTH = 'hatchback_images.npz'

# setting image shape to 32x32
IMG_WIDTH = 64
IMG_HEIGHT = 64

EPOCHS = 100
ACCELERATOR = "ddp"

GPUS = 1 # no of gpus
BATCH_SIZE = 64
LEARNING_RATE = 0.002
B1 = 0.5
B2 = 0.999
LATENT_DIM = 100
