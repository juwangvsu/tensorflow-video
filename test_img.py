import os
import time
#import cv2
import time
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from object_detection.utils import label_map_util as label_map_util
from object_detection.utils import visualization_utils as vis_util
#from matplotlib import pyplot as plt
from PIL import Image

PATH_TO_TEST_IMAGES_DIR = '/home/student/tensorflow/models-master/research/object_detection/test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

for image_path in TEST_IMAGE_PATHS:
    image = Image.open(image_path)
    #image_np = load_image_into_numpy_array(image)
    #plt.imshow(image_np)
    plt.imshow(image)
    #print(image.size, image_np.shape)
plt.show()

