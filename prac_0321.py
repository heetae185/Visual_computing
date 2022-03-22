import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img_bgr = cv.imread('prac_0316/sudoku.png')
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
img_gray = cv.imread('prac_0316/sudoku.png', cv.IMREAD_GRAYSCALE)

hist_r = cv.calcHist([img_bgr], [2], None, [256], [0, 255])
hist_g = cv.calcHist([img_bgr], [1], None, [256], [0, 255])
hist_b = cv.calcHist([img_bgr], [0], None, [256], [0, 255])
hist_gray = cv.calcHist([img_gray], [0], None, [256], [0, 255])   

ret_otsu, thresh1 = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY)

plt.subplot(1,3,1)
plt.imshow(img_gray, cmap='gray')
plt.subplot(1,3,2)
plt.plot(hist_gray, color='gray')
plt.subplot(1,3,3)
plt.imshow(thresh1, cmap='gray')
plt.show()

img = cv.imread('prac_0316/sudoku.png',0)
img = cv.medianBlur(img,5)  #주변 픽셀값에 의한 보정, noise 줄어드는 효과 (interpolation)
ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)  # 11은 11x11 pixel, 2는 보정을 위해서 뺌...?
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)
titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

img = cv.imread('prac_0321/wiki.jpg',0)
hist,bins = np.histogram(img.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()
plt.subplot(1,4,1)
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')

cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')

plt.subplot(1,4,2)
plt.imshow(img)

img2 = cdf[img]
plt.subplot(1,4,3)
plt.plot(cdf_normalized, color = 'b')
plt.hist(img2.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')

plt.subplot(1,4,4)
plt.imshow(img2)

plt.show()

img3 = cv.imread('prac_0321/wiki.jpg',0)
equ = cv.equalizeHist(img3)
res = np.hstack((img3,equ)) #stacking images side-by-side
cv.imwrite('res.png',res)

plt.imshow(res, cmap='gray')
plt.show()




