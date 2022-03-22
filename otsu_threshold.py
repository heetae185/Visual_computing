import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img1 = cv.imread('prac_0316/sudoku.png')
img = cv.imread('prac_0316/sudoku.png', cv.IMREAD_GRAYSCALE)
ret, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
ret, thresh2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
ret, thresh3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
ret, thresh4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
ret, thresh5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)

ret_otsu, thresh1_otsu = cv.threshold(img, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
ret_otsu, thresh2_otsu = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
ret_otsu, thresh3_otsu = cv.threshold(img, 127, 255, cv.THRESH_TRUNC | cv.THRESH_OTSU)
ret_otsu, thresh4_otsu = cv.threshold(img, 127, 255, cv.THRESH_TOZERO | cv.THRESH_OTSU)
ret_otsu, thresh5_otsu = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV | cv.THRESH_OTSU)

images = {'Original' : img, 'THRESH_BINARY' : thresh1, 'THRESH_BINARY_INV' : thresh2, 
          'THRESH_TRUNC' : thresh3, 'THRESH_TOZERO' : thresh4, 'THRESH_TOZERO_INV' : thresh5,
          'Original_OTSU' : img, 'THRESH_BINARY_OTSU' : thresh1_otsu, 'THRESH_BINARY_INV_OTSU' : thresh2_otsu, 
          'THRESH_TRUNC_OTSU' : thresh3_otsu, 'THRESH_TOZERO_OTSU' : thresh4_otsu, 'THRESH_TOZERO_INV_OTSU' : thresh5_otsu}

plt.figure(figsize=(9,12))
for i, (key, value) in enumerate(images.items()):
    plt.subplot(4,3,i+1), plt.imshow(value, 'gray', vmin=0, vmax=255)
    plt.title(key)
    plt.xticks([]), plt.yticks([])
    
plt.show()

print(ret, ret_otsu)    #127, OTSU값:96
print(img.shape)

#OTSU값 직접 구해보기
hist = cv.calcHist([img], [0], None, [255], [0, 255])
plt.plot(hist)
plt.show()
img_mean = np.mean(np.mean(img))
img_var = []
for i in range(hist.size):
    if i in (0, 254):   # 예외처리 : histogram이 2개로 쪼개지지 않는 경우
        hist_var = 0
        for j in range(hist.size):
            hist_var += (j * abs(hist[j] - img_mean)**2)
        hist_var /= hist.size    
        img_var.append(hist_var)
        continue
    h1 = hist[0:i+1]
    h2 = hist[i+1:256]
    h1_var = 0
    h2_var = 0
    for j1 in range(h1.size):
        h1_var += ((j1+1) * abs(h1[j1] - img_mean)**2)
    for j2 in range(h2.size):
        h2_var += ((j2+i+1) * abs(h2[j2] - img_mean)**2)
    h1_var /= h1.size
    h2_var /= h2.size    
    w1 = np.sum(h1/np.sum(hist))
    w2 = np.sum(h2/np.sum(hist))
    res_var = h1_var * w1 + h2_var * w2
    img_var.append(res_var)

otsu = img_var.index(min(img_var))  #직접 구해본 otsu값 : 96
#print(img_var)
plt.plot(img_var)
plt.show()
print(min(img_var), otsu)
#cv.imshow('1', img)
cv.waitKey(0)