import numpy as np
import cv2

img = cv2.imread('D:/Year4/FYP/Project/100_dpi.png')
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
t, rst = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY)
athdMEAN = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,10)
athdGAUS = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,10)
cropped = athdGAUS[920:1000,0:400]

cv2.imwrite('D:/Year4/FYP/Project/100dpi_1.jpg', athdGAUS)
cv2.imshow('image', cropped)#cv显示图片函数
cv2.waitKey(0)#窗口挂起
cv2.destroyAllWindows()#窗口销毁
print(athdGAUS.shape)