name=beauty_niweiya.mp4
python yolov7_track.py --yolo-weights weights/best.pt \
                --strong-sort-weights weights/osnet_ain_x1_0_imagenet.pth \
                --imgsz 768 \
                --conf-thres 0.1 \
                --save-vid \
                --save-txt \
                --source ../dataset/videos0715/${name}