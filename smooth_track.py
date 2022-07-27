# -*- encoding: utf-8 -*-
'''
@Description :   
@Author      :   suanyang 
@Time        :   2022/07/27 14:37:57
@Contact     :   suanyang@mininglamp.com
'''
import os 
import cv2
import sys
import random
import argparse
import numpy as np 
import pandas as pd

from pathlib import Path
from copy import deepcopy

sys.path.append('yolov7')
from yolov7.utils.plots import plot_one_box


def moving_average(interval, min_kernel_size=3, max_kernel_size = 99):
    kernel_size = len(interval) // 5 * 2 - 1  # can be 4\5\8, choose 5 
    
    if kernel_size < min_kernel_size:
        ans = np.ones(len(interval)) * int(np.average(interval))
    else:
        kernel_size = min(max_kernel_size, kernel_size)
        window = np.ones(int(kernel_size)) / float(kernel_size)
        re = np.convolve(interval, window, 'valid').astype(np.uint16)
        '''
        平滑后输出维度不等于输入维度，使用'same'参数，但仍存在边缘值问题，参考https://gemini-yang.blog.csdn.net/article/details/107176500
        因此，使用'valid'参数获取可用结果，然后对结果的边界进行等值填充
        填充范围为 (kernel_size-1) //2, kernel_size需要为奇数'''
        lb = np.ones((kernel_size-1)//2)*re[0]
        rb = np.ones((kernel_size-1)//2)*re[-1]
        ans = np.concatenate((lb, re, rb))
        # print(kernel_size, len(interval), len(ans))

    return ans

def change_bbox(det_dict, min_kernel_size=3, max_kernel_size = 99):
    # 已废弃
    smooth_dict = {}
    bboxs = []
    for key, data in det_dict.items():
        bboxs.append(data[-4:])
    bboxs = np.array(bboxs)
    for i in range(4):
        bboxs[:, i] = moving_average(bboxs[:, i], min_kernel_size=3, max_kernel_size = 99)
    for i, key in det_dict.keys():
        smooth_dict[key] = bboxs[i]
        
    return smooth_dict

def change_bbox_pandas(det_df, min_kernel_size=3, max_kernel_size = 99):
    # det_df为按照track_id 分组后的单一 track id
    smooth_dict = {}
    smooth_df = deepcopy(det_df).reset_index(drop=True)
    for col in ['x', 'y', 'w', 'h']:
        # print(len(np.array(det_df[col])), len(smooth_df[col]))
        smooth_df.drop(col, axis=1, inplace=True)
        size = len(det_df[col])
        smooth_npy = moving_average(np.array(det_df[col]), min_kernel_size=3, max_kernel_size = 99)
        smooth_dict[col] = smooth_npy
    
    smooth_df = pd.concat([smooth_df, pd.DataFrame(smooth_dict)], axis=1)
    return smooth_dict, smooth_df

def save_smooth_txt(txt_path, 
                    save_txt_path = None, 
                    txt_columns = ['video_id', 'frame_id', 'track_id', 'cls_id', 'conf', 'x', 'y', 'w', 'h', 'not_use']):
    if not save_txt_path:
        save_txt_path = os.path.splitext(txt_path)[0]+'_smooth.txt'
    
    df = pd.read_csv(txt_path, sep=' ', header=None)
    df.columns = txt_columns
    df.drop('not_use',axis=1, inplace=True)
    track_df = df.groupby('track_id', axis = 0, as_index = False)
    print(len(track_df))

    ans_df = pd.DataFrame(columns = txt_columns[:-1])
    for name, data in track_df:
        smooth_dict, smooth_df = change_bbox_pandas(data)
        ans_df = pd.concat([ans_df, smooth_df], axis = 0)

    ans_df.sort_values(by = ['video_id', 'frame_id'], inplace = True)
    ans_df.to_csv(save_txt_path, sep = ' ', header=None, index=False)

def draw_videos_by_txt(video_dir_name: str, 
                       video_name: str,
                       save_dir_name: str,
                       txt_path: str,
                       names :list = {0:'logo', 1:'ren', 2:'shiwu', 3:'wenzi'},
                       txt_columns = ['video_id', 'frame_id', 'track_id', 'cls_id', 'conf', 'x', 'y', 'w', 'h']):
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    df = pd.read_csv(txt_path, sep=' ', header=None)
    df.columns = txt_columns

    frame_index = 0
    print(os.path.join(video_dir_name, video_name))
    cap = cv2.VideoCapture(os.path.join(video_dir_name, video_name))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))# 获取视频宽度
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))# 获取视频高度
    fps = cap.get(cv2.CAP_PROP_FPS) #视频平均帧率
    print(fps, frame_width, frame_height)

    # 设置输出视频的参数，如果是灰度图，可以加上 isColor = 0 这个参数
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 设置输出视频为mp4格式
    write_video = cv2.VideoWriter(os.path.join(save_dir_name, os.path.splitext(video_name)[0]+'_smooth.mp4'),\
                                fourcc, fps, (frame_width,frame_height) ) #设置保存视频的名称和路径

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        
        frame_df = df.loc[df['frame_id'] == frame_index]
        frame_df.reset_index(drop=True, inplace=True)
        
        for i in range(len(frame_df)):
            x, y, w, h = frame_df['x'][i], frame_df['y'][i],frame_df['w'][i], frame_df['h'][i]
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)
            bboxes = [x1, y1, x2, y2]
            
            track_id = frame_df['track_id'][i]
            cls_id = frame_df['cls_id'][i]
            conf = frame_df['conf'][i]
            label = f"{track_id} {names[cls_id]}:{conf:.2f}"
            plot_one_box(bboxes, img, label=label, color=colors[cls_id], line_thickness=3)    
        write_video.write(img)
        frame_index += 1
    # 释放mp4    
    cap.release()  
    write_video.release() 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt-path', type=str, help='txt path for track results')
    parser.add_argument('--video-dir-name', type=str, default='../dataset/videos0601/', help='dirname for original video')
    parser.add_argument('--video-name', type=str, help='original video name')
    parser.add_argument('--save-dir-name', type=str, help='save dirname for video')
    args = parser.parse_args()

    txt_path = args.txt_path
    save_txt_path = os.path.splitext(txt_path)[0]+'_smooth.txt'
    video_name = args.video_name
    video_dir_name = args.video_dir_name
    save_dir_name = args.save_dir_name

    save_smooth_txt(txt_path)
    draw_videos_by_txt(video_dir_name,
                       video_name,
                       save_dir_name,
                       txt_path = save_txt_path)


    
