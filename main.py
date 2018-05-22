import cv2
import numpy as np
from matplotlib import pyplot as plt

# loading image
img0 = cv2.imread('melonLap1.jpg')

# converting to gray scale
gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

# remove noise
img = cv2.GaussianBlur(gray,(3,3),0)

# convolute with proper kernels
laplacian = cv2.Laplacian(img,cv2.CV_64F)

sobel = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)

canny = cv2.Canny(img,cv2.CV_64F,100,200)

# prewitt
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img_prewittx = cv2.filter2D(img, -1, kernelx)
img_prewitty = cv2.filter2D(img, -1, kernely)

# roberts
roberts_cross_v = np.array([[0,0,0],[0,1,0],[0,0,-1]])
roberts_cross_h = np.array([[0,0,0],[0,0,1],[ 0,-1,0]])
img_robertsx = cv2.filter2D(img, -1, roberts_cross_v)
img_robertsy = cv2.filter2D(img, -1, roberts_cross_h)

plt.subplot(3,3,1),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(3,3,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

plt.subplot(3,3,3),plt.imshow(sobel,cmap = 'gray')
plt.title('Sobel'), plt.xticks([]), plt.yticks([])

plt.subplot(3,3,4),plt.imshow(canny,cmap = 'gray')
plt.title('Canny'), plt.xticks([]), plt.yticks([])

plt.subplot(3,3,5),plt.imshow(img_prewittx + img_prewitty,cmap = 'gray')
plt.title('Prewit'), plt.xticks([]), plt.yticks([])

plt.subplot(3,3,6),plt.imshow(img_robertsx + img_robertsy,cmap = 'gray')
plt.title('Roberts'), plt.xticks([]), plt.yticks([])

plt.show()
