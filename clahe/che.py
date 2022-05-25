import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = []
plt.figure(figsize=(12,10))
for i in range(7):
    img.append(cv.imread('color_histogram_equal/low_contrast'+str(i+1)+'.jpg'))
    hist_r = cv.calcHist([img[i]], [2], None, [255], [0,255])
    hist_g = cv.calcHist([img[i]], [1], None, [255], [0,255])
    hist_b = cv.calcHist([img[i]], [0], None, [255], [0,255])

    plt.subplot(2,4,i+1)
    plt.plot(hist_r, color='r')
    plt.plot(hist_g, color='g')
    plt.plot(hist_b, color='b')

plt.figure(figsize=(12,10))
for i in range(7):
    img[i] = cv.cvtColor(img[i], cv.COLOR_BGR2RGB)
    plt.subplot(2,4,i+1)
    plt.imshow(img[i])
plt.show()
plt.show()

hist_r,bins_r = np.histogram(img[4][:,:,0].flatten(),256,[0,256])
hist_g,bins_g = np.histogram(img[4][:,:,1].flatten(),256,[0,256])
hist_b,bins_b = np.histogram(img[4][:,:,2].flatten(),256,[0,256])
cdf_r = hist_r.cumsum()
cdf_g = hist_g.cumsum()
cdf_b = hist_b.cumsum()
cdf_r_normalized = cdf_r * float(hist_r.max()/cdf_r.max())
cdf_g_normalized = cdf_g * float(hist_g.max()/cdf_g.max())
cdf_b_normalized = cdf_b * float(hist_b.max()/cdf_b.max())
hist_color=['r', 'g', 'b']
for i in range(3):
    plt.subplot(2,4,i+1)
    plt.plot(cdf_r_normalized, color='black')
    plt.hist(img[4][:,:,i].flatten(), 256, [0,256], color=hist_color[i])
    plt.xlim([0,256])
plt.subplot(2,4,4)
plt.imshow(img[4])

def cdf_m_maker(cdf):
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    return cdf

cdf_r = cdf_m_maker(cdf_r)
cdf_g = cdf_m_maker(cdf_g)
cdf_b = cdf_m_maker(cdf_b)

img2_r = cdf_r[img[4][:,:,0]]
img2_g = cdf_g[img[4][:,:,1]]
img2_b = cdf_b[img[4][:,:,2]]
img2 = cv.merge((img2_r, img2_g, img2_b))

hist2_r,bins_r = np.histogram(img2[:,:,0].flatten(),256,[0,256])
hist2_g,bins_g = np.histogram(img2[:,:,1].flatten(),256,[0,256])
hist2_b,bins_b = np.histogram(img2[:,:,2].flatten(),256,[0,256])
cdf2_r = hist2_r.cumsum()
cdf2_g = hist2_g.cumsum()
cdf2_b = hist2_b.cumsum()
cdf2_r_normalized = cdf2_r * float(hist2_r.max()/cdf2_r.max())
cdf2_g_normalized = cdf2_g * float(hist2_g.max()/cdf2_g.max())
cdf2_b_normalized = cdf2_b * float(hist2_b.max()/cdf2_b.max())

for i in range(3):
    plt.subplot(2,4,i+5)
    plt.plot(cdf2_r_normalized, color='black')
    plt.hist(img2[:,:,i].flatten(), 256, [0,256], color=hist_color[i])
plt.subplot(2,4,8)
plt.imshow(img2)

plt.show()

    