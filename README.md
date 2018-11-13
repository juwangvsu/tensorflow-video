# Deep-Learning
# jwang note 7/9/18

----------11/12/2018 model/label option to use frozen graph  ------
./obj_video_droneimg.sh -d p3dx_frozen -l toy_label_map.pbtxt

files:
	tensorflow video/obj_video_droneimg.py
	tensorflow video/obj_video_droneimg.sh

--------- 11/9/18  retrained inception model and test -----
TBD

--------- 11/7/18 export retrained model to frozen -----
ub16_tensorflow
~/bin/tf_export.sh
	this will create a frozen pb file from a checkpoint (typically in train
		directory)
		it generate a folder in object_detection folder
	must run from ~/tensorflow/models-master/research
	create a link: for access data with relative path
		ln -s /media/student/code1/tensorflow\ video/tensorflow_p3dx_detector tensorflow_p3dx_detector
	then edit accordingly
	INPUT_TYPE=image_tensor
	PIPELINE_CONFIG_PATH=ssd_inception_v2_coco_2018_01_28/pipeline2.config
	TRAINED_CKPT_PREFIX=ssd_inception_v2_coco_2018_01_28/model.ckpt
	EXPORT_DIR=object_detection/inception_frozen
	python object_detection/export_inference_graph.py \
    		--input_type=${INPUT_TYPE} \
    		--pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    		--trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    		--output_directory=${EXPORT_DIR}

results and issues:
	the frozen model from p3dx also work with the obj_detect_node.py
	but it is much slower than the faster_rcnn frozen model
	inception is faster than faster_rcnn, says model zoo, confirmed
	mobilenet also fast, but export problem.
	the label file does not affect the detection, just the labeling part.
		so we can just edit the label file to make a quick hack:
		edit label entry for "traffic light" to "p3dx" will make 
		faster_rcnn appearing to be trained to detect p3dx.

exported frozen graph:
	~/tensorflow/.../object_detection/data
					/inception_frozen
					/p3dx_frozen
					/ssd_mobilenet_v2_coco_2018_03_29

--------- 11/6/18 spin off obj_video_droneimg.py to ros node-----
	-- create obj_detect_node.py	
	-- to detect and display rostopic image
	-- the node will be placed under rqt_mypkg
	-- make sure only initiate session one time, don't do it for
		each image. otherwise overhead high and slow, see code
		    with detection_graph.as_default():
		        with tf.Session(graph=detection_graph) as sess:
           			while not rospy.is_shutdown():
		the while loop must be inside.
	-- rosrun rqt_mypkg obj_detect_node.py
	it only need two model related files: the frozen model .pb file
		and the mscoco_label_map.pbtxt

-----------10/30/18 tensorflow detection retrain existing model -----------------
see 7/9/18 retrain note for general testing procedure.
this note start a new retrain using ardrone1 frames containing p3dx.
	(1) prepare training data set and eval data set.
		put all jpg images to folder "images"
		(I) annotations:
		mkdir annotations. 
		cd ~/labelImg, ./labelImg, click "change save dir" and select
			"annotations" folder under /media/student/code1/
			tensorflow video/tensorflow_p3dx_detector
			click "open dir" to the "images" folder
			then go through all images create box around objs,
			in this example we use two new labels : "p3dx" and
			"drone"
		(II) create file: toy_label_map.pbtxt
		(III) create file: tensorflow_p3dx_detector/trainval.txt\
		(IV) convert data record:
			python create_pet_tf_record.py
			this will generate pet_train.record 
					   pet_val.record
		(V) run training:
			(V.1) download a frozen model, and
			copy frozen model's model.ckpt.* to the train/
			(V.2) edit .config file accordinly
		        python train.py   --logtostderr -pipeline_config_path=\
			faster_rcnn_resnet101_coco.config     --train_dir=train
			(V.3) do not mix .config and ckpt files from different
				model. the pipeline.config from the frozen model
				is edited.
				the mobilenet config pipeline 
				is still being tested, ub16_tensorflow, hpelite
			(V.4) the training error after several training steps
			step 10: loss = 11.6917 (0.412 sec/step)
			step 11: loss = 11.7478 (0.413 sec/step)
			: Incompatible shapes: [2,1917] vs. [3,1]
	
		(VI) run evaluation: see (b)
		
old summery for quick testing steps:
	(a) cd /media/.../tensorflow_toy_detector
	(b) python eval.py --logtostderr --pipeline_config_path=\
faster_rcnn_resnet101_coco.config --checkpoint_dir=train --eval_dir=eval
	(c) @another terminal, 
    		tensorboard --logdir=eval
	(d) view results: http://localhost:6006/#graphs&run=.


----------10/28/2018 extract image , head, pose from rosbag, obj detection -----
/media/student/code1/tensorflow video
	(1) python obj_video_droneimg.sh ardrone1 
		-this script extract frames from bag, then run cnn to detect obj
		the ardrone1/image_raw/ is where testing image are saved.
		the results are frame0143.jpg_detect.jpg
		and ardrone1/ardrone1_det_rst.txt
	(2) rosbag/extract_pose_etc.sh
		- this extract pose info
files: code
	(1.1)./extract_frames_frombag.sh
		--- this extract frames from bag
	(1.2)obj_video_droneimg.py
		--- use pretrained tensorflow model to detect obj and 
		generate detection result file.
	(1.1.1)image_extract_frombag.launch 
		-> /media/student/code2/rosbag/image_extract_frombag.launch
	(2.1) rosbag/imagetopic_head_frombag.launch
	(2.1.1) imagetopic_head.py
		 -> /home/student/turtlebot/src/rqt_mypkg/scripts/imagetopic_head.py

files: data
rosbag/ardrone1_img_head.txt     rosbag/mavros2_global_local.txt
rosbag/ardrone2_img_head.txt     rosbag/mavros2_local_pose.txt
rosbag/mavros1_global_local.txt  rosbag/mavros1_local_pose.txt
rosbag/ardrone1/ardrone1_img_head.txt 
rosbag/ardrone2/ardrone2_img_head.txt 

-----------------10/27/18 extrac frames and encode to video from bag---------
	jpg_to_mp4.sh 
		-> /media/student/code2/rosbag/jpg_to_mp4.sh

-----------------10/27/18 get a rosbag from gazebo drone img topic---------
@new amd computer, ub14 current partition:
	px4simsilt-1.5.5...a
	add something you want to video record on gazebo
	use joystick to fly
	rosbag record

---------10/27/2018 extract frame from mp4 video-----

ffmpeg -i p3dx_video.mp4 frames%d.jpg
mkdir images; mv frame* images
to select some frames for training, (such as every 10 frames), create a list
(using excel) file.txt, then:
mkdir new_folder
for file in $(<file.txt); do cp "$file" new_folder; done

-----------10/27/18 ub16 tensorflow, object_detection models, install tips -----------------------
these are two installation process for ub16:
   tensorflow:
	sudo apt-get install libcupti-dev
	wget https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip
	unzip protoc-3.3.0-linux-x86_64.zip  -d protoc330
		add ~/protoc330/bin/ to .bashrc PATH
	sudo apt install python-dev python-pip
	sudo pip install --ignore-installed enum34
	pip install --upgrade tensorflow
	python -c "import tensorflow as tf; print(tf.__version__)"
	sudo pip install moviepy
	sudo pip install requests

   object_detection models:
	https://github.com/tensorflow/models.git (or download zip)
	pip install --user Cython
	pip install --user contextlib2
	pip install --user pillow
	pip install --user lxml
	pip install --user jupyter
	pip install --user matplotlib	
	git clone https://github.com/cocodataset/cocoapi.git
	cd cocoapi/PythonAPI
	make
	cp -r pycocotools <path_to_tensorflow>/models/research/
	assuming installed at ~/tensorflow/models/research/
		.bashrc
		export PYTHONPATH=$PYTHONPATH:~/tensorflow/models/research/:~/tensorflow/models/research/slim
		or
		export PYTHONPATH=$PYTHONPATH:~/tensorflow/models-master/research/::~/tensorflow/models-master/research/slim
	python object_detection/builders/model_builder_test.py
	
-----------7/9/18 jupyter notebook tensorflow video -------------------------
Try notebook in /media.../code1/tensorflow-video
See webpage: https://towardsdatascience.com/is-google-tensorflow-object-detection-api-the-easiest-way-to-implement-image-recognition-a8bd1f500ea0
https://github.com/priya-dwivedi/Deep-Learning

To run the notebook, /usr/bin/pip install moviepy
open terminal, cd /media .../tensorflow-video/; jupyter notebook; 
A webpage will open and navigate to  Object_Detection_Tensorflow_API.ipynb
The notebook run cell by cell, several paths must to fixed to point to the files in my object_detection folder , if a cell contain error, you can fix and rerun that cell, then continue the rest of cells.
youtube-dl https://www.youtube.com/watch?v=ytuCzChCSs4
Change to video1.mp4	

-----------7/9/18 tensorflow toy detection retrain from existing model -------------------------
https://towardsdatascience.com/building-a-toy-detector-with-tensorflow-object-detection-api-63c0fdf2ac95
https://github.com/priya-dwivedi/Deep-Learning
Toy detection example:
/media/student/code1/tensorflow-video/Deep-Learning/tensorflow_toy_detector
The stuff from github is not complete, specifically, it does not have .config file and the 
trained model
Trained Models 
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
	Ssd_mobilenet_v2_coco_2018_03_29.tar.gz
		Model.ckpt,frozen_inference_graph.pb
Test run toy detection: 
(1) Download the trained model: faster_rcnn_resnet101_coco_2018_01_28.tar.gz
	http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz

	/media/student/code1/faster_rcnn_resnet101_coco_2018_01_28
(2) Cp ~/tensorflow-sample/models/research/object_detection/samples/
	configs/faster_rcnn_resnet101_coco.config to 
	/media/student/code1/tensorflow-video/Deep-Learning/tensorflow_toy_detector

(3) Edit faster_rcnn_resnet101_coco.config to fix paths and num_examples: 8 to limit the run time of eval. Note the eval.py loop forever, runs num_examples each time and wait some time. It keep checking new checkpoint files for evaluation. Each example requires 4 second for this setup (y520)

(4) cd /media/student/code1/tensorflow-video/Deep-Learning/tensorflow_toy_detector$
     for later version of object_detection, need to edit train.py as train.py.new
	to contain this "from object_detection.legacy import trainer",
	same for eval.py
     Run train:
	python train.py     --logtostderr -pipeline_config_path=\
faster_rcnn_resnet101_coco.config     --train_dir=train
     Run eval:
	the eval.py must be modified to change input_config to input_config[0] at
		several place, on the 10/27/18 installation on ub16, as eval.py.new

	python eval.py --logtostderr --pipeline_config_path=\
faster_rcnn_resnet101_coco.config --checkpoint_dir=train --eval_dir=eval
(5)

Run tensorboard to view results:
    tensorboard --logdir=eval

http://localhost:6006/#graphs&run=.

This repository contains deep learning related projects I did as part of Kaggle submissions or for Udacity's Deep Learning Foundations Nano Degree.

Projects Outline

* Kaggle Fisheries - Keras convolution neural network developed to predict fish types for Kaggle competition
https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring

* Image Classification - CIFAR
Work done as part of Udacity's deep learning foundations nano degree
