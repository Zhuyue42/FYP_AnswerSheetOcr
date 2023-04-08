import cv2
import numpy as np

img = cv2.imread('D:\Year4\FYP\Project\dpi100_4.png')
median = cv2.medianBlur(img,5)
cv2.imwrite('D:\Year4\FYP\Project\dpi100_5.png',median)