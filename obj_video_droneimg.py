import os
import sys
import cv2
import time
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from object_detection.utils import label_map_util as label_map_util
from object_detection.utils import visualization_utils as vis_util
CWD_PATH = os.getcwd()
CWD_PATH="/home/student/tensorflow/models-master/research"
print("jwang")
print (CWD_PATH)

# Path to frozen detection graph. This is the actual model that is used for the object detection.
#MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
#MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
#MODEL_NAME = 'inception_frozen'
#MODEL_NAME = 'p3dx_frozen'
MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
LABEL_NAME = 'mscoco_label_map.pbtxt'
LABEL_NAME = 'toy_label_map.pbtxt'
print(len(sys.argv))
if len(sys.argv)>2:
	# two parameter, the 1st is the image folder, 2nd is network model
	MODEL_NAME = sys.argv[2] 
if len(sys.argv)>3:
	# three parameter, the 1st is the image folder, 2nd is network model
	# 3rd label
	LABEL_NAME = sys.argv[3] 

PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', LABEL_NAME)

NUM_CLASSES = 90

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#this one return processed img, this is used in process_image() 
def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    return image_np


#this one return all relevent information
def detect_objects_2(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    return image_np, boxes, scores, classes

# First test on images
PATH_TO_TEST_IMAGES_DIR0 = '/media/student/code1/tensorflow video/'
PATH_TO_TEST_IMAGES_DIR = '/media/student/code1/tensorflow video/'+sys.argv[1]
#argv[1] is the name of subfolder to the images, ardrone1 or ardrone2

#PATH_TO_TEST_IMAGES_DIR = '/home/student/tensorflow/models-master/research/object_detection/test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image_raw/frame{}.jpg'.format(str(i).zfill(4))) for i in range(70, 200) ]
ardrone1_det_rstfn = os.path.join(PATH_TO_TEST_IMAGES_DIR, sys.argv[1]+'_det_rst.txt')
ardrone1_rstfile = open(ardrone1_det_rstfn, "wb")

#camera_info from rosbag: height: 480 width: 640
#project matrix: P: [205.4, 0, 320.5, -14.3
# 0, 205.4, 240.5, 0.0
# 0, 0, 1.0, 0]
# boxes normalized coord in img frame, img frame origin assumed left up conner


ardrone1_rstfile.write('boxes normalized coord in img frame, img frame origin assumed left up conner')
ardrone1_rstfile.write('camera_info from rosbag: height: 480 width: 640 project matrix: P: [205.4, 0, 320.5, -14.3; 0, 205.4, 240.5, 0.0 ; 0, 0, 1.0, 0]\n')

ardrone1_rstfile.write('filename')
ardrone1_rstfile.write("\t")
ardrone1_rstfile.write("ymin")
ardrone1_rstfile.write("\t")
ardrone1_rstfile.write("xmin")
ardrone1_rstfile.write("\t")
ardrone1_rstfile.write("ymax")
ardrone1_rstfile.write("\t")
ardrone1_rstfile.write("xmax")
ardrone1_rstfile.write("\t")
ardrone1_rstfile.write("score")
ardrone1_rstfile.write("\t")
ardrone1_rstfile.write("class")
ardrone1_rstfile.write("\n")

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
from PIL import Image
def loadimg():
  for image_path in TEST_IMAGE_PATHS:
    image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image)
    #plt.imshow(image_np)
    #plt.imshow(image)
    print(image.size, image_np.shape)

#Load a frozen TF model 
print(PATH_TO_CKPT)
detection_graph = tf.Graph()
print(detection_graph)
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
plt.figure(figsize=IMAGE_SIZE)
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        for image_path in TEST_IMAGE_PATHS:
            image = Image.open(image_path)
            image_np = load_image_into_numpy_array(image)
            image_process,boxes, scores, classes = detect_objects_2(image_np, sess, detection_graph)
	    img = Image.fromarray(image_process, 'RGB') 
	    #img.save(image_path+'_detect.jpg') 
            print("show processed img for "+image_path)
	    print(boxes[0][0])
	    print(scores[0][0])
	    print(classes[0][0])
	    ardrone1_rstfile.write(image_path[::-1][:13][::-1])
	    ardrone1_rstfile.write("\t")
	    ardrone1_rstfile.write(str(boxes[0][0]))
	    ardrone1_rstfile.write("\t")
	    ardrone1_rstfile.write(str(scores[0][0]))
	    ardrone1_rstfile.write("\t")
	    ardrone1_rstfile.write(str(classes[0][0]))
	    ardrone1_rstfile.write("\n")
            print(image_process.shape)
            #plt.figure(figsize=IMAGE_SIZE)
            #plt.imshow(image_process)
	    #plt.show(block=False)
	    cv2.imshow('cv_img_detected', image_process)
            cv2.waitKey(5)

ardrone1_rstfile.close()
plt.show()
#plt.show(block=False)
from moviepy.editor import VideoFileClip
def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # you should return the final output (image with lines are drawn on lanes)
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_process = detect_objects(image, sess, detection_graph)
            return image_process
