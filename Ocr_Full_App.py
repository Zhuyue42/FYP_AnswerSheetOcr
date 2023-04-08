import tkinter
from tkinter import filedialog
import tkinter.font as tf
import numpy as np
import cv2
import pytesseract
from PIL import Image

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
    screenCnt=None
    boo=False
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
 
    # res = cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    # res = cv2.drawContours(image, cnts[0], -1, (0, 255, 0), 2)
    # show(orig)
    return orig, ratio, screenCnt,boo

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

def upload_file():
    # askopenfilename 上传1个;askopenfilenames上传多个
    result_text.delete(1.0, 'end')
    pytesseract.pytesseract.tesseract_cmd = "D://Tesseract-Ocr5.0//tesseract.exe"
    select_file = tkinter.filedialog.askopenfilename()

    orig, ratio, screenCnt,boo = edge_detection(select_file)
     # screenCnt 为四个顶点的坐标值，但是我们这里需要将图像还原，即乘以以前的比率
    # 透视变换  这里我们需要将变换后的点还原到原始坐标里面
    if boo==True:
        warped = four_point_transform(orig, screenCnt.reshape(4, 2)*ratio)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        thresh=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,10)
        thresh_resize= resize(thresh,width=675 ,height = 900)
        cropped = thresh_resize[760:850,0:330]
        cv2.imwrite('D:\Year4\FYP\Project\A4.jpg', cropped)
        text = pytesseract.image_to_string('D:\Year4\FYP\Project\A4.jpg')

        result_text.insert("insert", f'Identify Results:{text}')
    else:
        gray1=cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        thresh1=cv2.adaptiveThreshold(gray1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,10)
        cropped1=thresh1[920:1000,0:400]
        cv2.imwrite('D:\Year4\FYP\Project\dpi100_4.png',cropped1)
        text1 = pytesseract.image_to_string('D:\Year4\FYP\Project\dpi100_4.png')
        #test1_s=text1.split()
        print(type(text1))
        result_text.insert("insert", f'Identify Results:{text1}')

    
    
    
    

if __name__ == '__main__':
    root = tkinter.Tk()
    root.title('Answer Sheet')
    root.minsize(800, 600)
    my_font = tf.Font(family='微软雅黑', size=15)  # 设置字体

    main_frame = tkinter.Frame(root).grid()
    # 选择文件按钮
    choice_file_btn = tkinter.Button(main_frame, text='请选择文件', command=upload_file)
    # 计算结果显示框
    result_text = tkinter.Text(main_frame, width=35, height=20, font=my_font)

    choice_file_btn.grid(row=0, column=0)
    result_text.grid(row=0, column=1)

    root.mainloop()
