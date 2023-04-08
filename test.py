from re import T
import cv2
import numpy as np
from PIL import Image
import os, sys
import tkinter
from tkinter import filedialog
import tkinter.font as tf
os.chdir(sys.path[0])

boo=True

def show(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w*r), height)
    else:
        r = width / float(w)
        dim = (width, int(h*r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

def edge_detection(img_path):
    # 读取输入
    img = cv2.imread(img_path)
    # 坐标也会相同变换
    ratio = img.shape[0] / 500.0
    orig = img.copy()
 
    image = resize(orig, height=500)
    # 预处理，将图片转为黑白二值图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 75, 200)
    #show(edged)

    # *************  轮廓检测 ****************
    # 轮廓检测
    #, hierarchy
    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
 
    # 遍历轮廓
    for c in cnts:
        # 计算轮廓近似
        peri = cv2.arcLength(c, True)
        # c表示输入的点集，epsilon表示从原始轮廓到近似轮廓的最大距离，它是一个准确度参数
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
 
        # 4个点的时候就拿出来
        if len(approx) == 4:
            screenCnt = approx
            boo=True
            break
        else:
            boo=False
            break
 #猜想，无背景时候的screenCnt可能没有值，所以会有报错
 
    if boo==True:
        res = cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
        print([screenCnt])
        show(res)
    else:
        print("The image does not have background")

edge_detection("D:/Year4/FYP/Project/AnswerSheet.jpg")
# if __name__ == '__main__':
#     edge_detection("D:/Year4/FYP/Project/AnswerSheet.jpg")
