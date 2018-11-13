#usage: ./obj_video_droneimg.sh [extract]
#extrac frames (default skip) and run obj_detect cnn on ardrone1 and ardrone2

do_detect()
{
#$1 model
	echo "do detect $1 $2"
	rm ardrone1/image_raw/*detect.jpg
	python obj_video_droneimg.py ardrone1 $1 $2
	return
	rm ardrone2/image_raw/*detect.jpg
	python obj_video_droneimg.py ardrone2 $1 $2
}
do_extract()
{
	echo "do extract"
	pushd .
	cd /media/student/code2/rosbag
#	./extract_frames_frombag.sh
	popd
}
echo "usage: ./obj_video_droneimg.sh [-edlh]"
dflag=false
label=""
while getopts "h?ed:l:" opt; do
    case "$opt" in
    h|\?)
	echo "usage: ./obj_video_droneimg.sh -eh"
	echo "usage: ./obj_video_droneimg.sh -d modelname [-l label]"
	echo" modelname: ssd_mobilenet_v2_coco_2018_03_29"
	echo" 		 p3dx_frozen"
	echo" 		 inception_frozen"
	echo" label: 	mscoco_label_map.pbtxt"
	echo" 		 toy_label_map.pbtxt"
        exit 0
        ;;
    e)  
	do_extract
        ;;
    l)  
	label=$OPTARG
        ;;
    d)  
	model=$OPTARG
	echo $model
	dflag=true
#	do_detect $model
        ;;
    esac
done
echo "$dflag"
if $dflag ; then
	do_detect $model $label
fi

