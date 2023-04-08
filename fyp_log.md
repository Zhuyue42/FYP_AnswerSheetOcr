##2022/3/5
成功在python环境下使用tesseract识别，解决两个问题。
pillow中的image.open()中必须填写绝对路径，相对路径无法识别。
tesseract_cmd后面也必须填写tesseract.exe的绝对路径。
![tesseract_cmd](./log_image/tesseract_cmd.png)
##2022/3/8
查找到如何使用opencv对图片进行透视矫正与灰度处理。图片预处理应分为三步，透视矫正，灰度处理，切割。
##20222/4/3
解决无法使用文件相对位置的难题。在程序开始前添加代码
```python
import os, sys
os.chdir(sys.path[0])

def edge_detection(img_path):
    # 读取输入读入格式为BGR
    img = cv2.imread(img_path)
    # 坐标也会相同变换
    #img.shaoe[0],返回图片垂直高度
    ratio = img.shape[0] / 500.0#储存此时高度与500的比率，方便还原
    orig = img.copy()
 
    image = resize(orig, height=500)#将图片高度更改为500
    # 预处理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#转为unit8灰度图
    blur = cv2.GaussianBlur(gray, (5, 5), 0)#高斯矩阵长宽为5，标准差为0
    edged = cv2.Canny(blur, 75, 200)#处理过程的两个阈值
    show(edged)
```
##2022/4/6
查找到的opencv检测代码在图片无背景色时效果很差。解决方法：添加与背景色检测或在图形界面分两个入口。

##2022/4/8
对变量screenCnt添加判断，判断图片是否有背景。
##2022/4/12
实现前端窗口传入图片可以直接识别，但是缺失opencv处理环节
##2022/4/14
添加了对图片是否有背景的判断，后续可以分为两条路
##2022/4/15
```python
import cv2
import numpy as np
from PIL import Image
import pytesseract
 
 
def show(image):
    cv2.imshow('image', image)#cv显示图片函数
    cv2.waitKey(0)#窗口挂起
    cv2.destroyAllWindows()#窗口销毁
 
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None#初始化变量
    (h, w) = image.shape[:2]#image是使用opencv读入的图片，以numpy形式储存
    #不传入指定参数，直接返回原图
    if width is None and height is None:
        return image
    #没有传入指定宽度
    if width is None:
        r = height / float(h)
        dim = (int(w*r), height)
    else:
        r = width / float(w)
        dim = (width, int(h*r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized
 
 
def edge_detection(img_path):
    # *********  预处理 ****************
    # 读取输入
    img = cv2.imread(img_path)
    # 坐标也会相同变换
    ratio = img.shape[0] / 500.0
    orig = img.copy()
 
    image = resize(orig, height=500)
    # 预处理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 75, 200)
 
    # *************  轮廓检测 ****************
    # 轮廓检测
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
            break
 
    # res = cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    # res = cv2.drawContours(image, cnts[0], -1, (0, 255, 0), 2)
    # show(orig)
    return orig, ratio, screenCnt
 
 
def order_points(pts):
    # 一共四个坐标点
    rect = np.zeros((4, 2), dtype='float32')
     
    # 按顺序找到对应的坐标0123 分别是左上，右上，右下，左下
    # 计算左上，由下
    # numpy.argmax(array, axis) 用于返回一个numpy数组中最大值的索引值
    s = pts.sum(axis=1)  # [2815.2   1224.    2555.712 3902.112]
    print(s)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
 
    # 计算右上和左
    # np.diff()  沿着指定轴计算第N维的离散差值  后者-前者
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect
 
 
# 透视变换
def four_point_transform(image, pts):
    # 获取输入坐标点
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
 
    # 计算输入的w和h的值
    widthA = np.sqrt(((br[0] - bl[0])**2) + ((br[1] - bl[1])**2))
    widthB = np.sqrt(((tr[0] - tl[0])**2) + ((tr[1] - tl[1])**2))
    maxWidth = max(int(widthA), int(widthB))
 
    heightA = np.sqrt(((tr[0] - br[0])**2) + ((tr[1] - br[1])**2))
    heightB = np.sqrt(((tl[0] - bl[0])**2) + ((tl[1] - bl[1])**2))
    maxHeight = max(int(heightA), int(heightB))
 
    # 变化后对应坐标位置
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]],
        dtype='float32')   
 
    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
    # 返回变换后的结果
    return warped
 
 
# 对透视变换结果进行处理
def get_image_processingResult():
    img_path = 'images/receipt.jpg'
    orig, ratio, screenCnt = edge_detection(img_path)
    # screenCnt 为四个顶点的坐标值，但是我们这里需要将图像还原，即乘以以前的比率
    # 透视变换  这里我们需要将变换后的点还原到原始坐标里面
    warped = four_point_transform(orig, screenCnt.reshape(4, 2)*ratio)
    # 二值处理
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
 
    cv2.imwrite('scan.jpg', thresh)
 
    thresh_resize = resize(thresh, height = 400)
    # show(thresh_resize)
    return thresh
 
 
 
def ocr_recognition(filename='tes.jpg'):
    img = Image.open(filename)
    text = pytesseract.image_to_string(img)
    print(text)
 
 
if __name__ == '__main__':
    # 获取矫正之后的图片
    # get_image_processingResult()
    # 进行OCR文字识别
    ocr_recognition()
```
##2022/4/16
直接使用原作者的二值化函数会因为阴影问题导致图片无法识别，问题已解决，换用另一种二值化方式。

##2022/4/17
裁剪参数已找到，组合代码架构已理清，明天开始拼装，希望后天拼装完成。然后进行手写数字和字母的训练，预计一周完成。开始着手毕业论文，一天进度一页。