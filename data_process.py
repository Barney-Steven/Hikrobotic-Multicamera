#以 utf-8 的编码格式打开指定文件
import string
import re
import os
from unittest import result
from cv2 import line
import numpy

#读取文件获取数据
f = open("face_detection.txt",encoding = "utf-8")
lines = f.readlines()
f.close()

def face_detection(lines, ret):
    #摄像头标签
    label_id = lines[0].split(": ")[1]
    label_id = int(label_id)

    #置信度
    current_socre = lines[1].split(": ")[1]
    current_socre = float(current_socre)
    current_socre = round(current_socre, 4)

    #bounding box xmin ymin weight height
    bb = [0 for x in range(0,4)]
    for i in range(5, 9):
        a = i - 5
        bb[a] = lines[i].split(": ")[1]
        bb[a] = float(bb[a])

    #right eye 11 12 / 0 1
    #left eye 15 16 / 2 3
    #nose tip 19 20 / 4 5
    #mouth center 23 24 / 6 7
    #right ear tragion 27 28 / 8 9
    #left ear tragion 31 32 / 10 11
    face_feature = [0 for x in range(0,12)]
    for i in range(0, 6):
        a = 11 + i * 4
        b = a + 1
        face_feature[2*i] = float(lines[a].split(": ")[1])
        face_feature[2*i+1] = float(lines[b].split(": ")[1])
    if ret == 1:
        return label_id
    elif ret == 2:
        return current_socre
    elif ret == 3:
        return bb
    elif ret == 4:
        return face_feature
for i in range(1, 5):
    result =  face_detection(lines, i)
    print(result)