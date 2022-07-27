name=beauty_libai
saveDir=${name}_dist002_iou03
python my_yolov7_track.py --yolo-weights weights/region_head_best.pt \
                --strong-sort-weights weights/osnet_ain_x1_0_imagenet.pth \
                --imgsz 960 \
                --conf-thres 0.1 \
                --iou-thres 0.65 \
                --strong-cfg-custom \
                --strong-max-dist 0.1 \
                --strong-iou 0.3 \
                --name ${saveDir} \
                --half \
                --save-vid \
                --save-txt \
                --exist-ok \
                --source ../dataset/videos0715/${name}.mp4

python smooth_track.py --txt-path ./runs/${saveDir}/${name}.txt \
                       --video-dir-name ../dataset/videos0715/ \
                       --video-name ${name}.mp4 \
                       --save-dir-name ./runs/${saveDir}/