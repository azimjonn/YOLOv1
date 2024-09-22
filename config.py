# patch size
S = 7

# number of predictions per patch
B = 2

# number of classes
C = 80

# training image Size
IMAGE_SIZE = (448, 448)
COCO_ROOT = '/data/coco/'

# training hyperparameters
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-4
NUM_WORKERS = 8

# loss coefficients
LAMBDA_COORD = 5
LAMBDA_NOOBJ = 0.5