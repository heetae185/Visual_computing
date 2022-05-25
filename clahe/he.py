import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('low_contrast5_3.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)    #이미지를 RGB로 변경
hist_color=['r', 'g', 'b']

#cdf그래프를 그리기 위한 과정
hist_r,bins_r = np.histogram(img[:,:,0].flatten(),256,[0,256])
hist_g,bins_g = np.histogram(img[:,:,1].flatten(),256,[0,256])
hist_b,bins_b = np.histogram(img[:,:,2].flatten(),256,[0,256])
cdf_r = hist_r.cumsum()
cdf_g = hist_g.cumsum()
cdf_b = hist_b.cumsum()
cdf_r_normalized = cdf_r * float(hist_r.max()/cdf_r.max())
cdf_g_normalized = cdf_g * float(hist_g.max()/cdf_g.max())
cdf_b_normalized = cdf_b * float(hist_b.max()/cdf_b.max())
cdf_normalized = [cdf_r_normalized, cdf_g_normalized, cdf_b_normalized]

plt.figure(figsize=(12,8))
for i in range(3):
    plt.subplot(2,4,i+1)
    plt.plot(cdf_normalized[i], color='black')
    plt.hist(img[:,:,i].flatten(), 256, [0,256], color=hist_color[i])
    plt.xlim([0,256])
plt.subplot(2,4,4)
plt.imshow(img)

#cdf에 따라 Histogram Equalize 시키는 함수
def cdf_m_maker(cdf):
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    return cdf

cdf_r = cdf_m_maker(cdf_r)
cdf_g = cdf_m_maker(cdf_g)
cdf_b = cdf_m_maker(cdf_b)

#Histogram Equalize된 이미지 img2에 저장
img2_r = cdf_r[img[:,:,0]]
img2_g = cdf_g[img[:,:,1]]
img2_b = cdf_b[img[:,:,2]]
img2 = cv.merge((img2_r, img2_g, img2_b))

#HE적용된 이미지 히스토그램에 대한 cdf그리기
hist2_r,bins_r = np.histogram(img2[:,:,0].flatten(),256,[0,256])
hist2_g,bins_g = np.histogram(img2[:,:,1].flatten(),256,[0,256])
hist2_b,bins_b = np.histogram(img2[:,:,2].flatten(),256,[0,256])
cdf2_r = hist2_r.cumsum()
cdf2_g = hist2_g.cumsum()
cdf2_b = hist2_b.cumsum()
cdf2_r_normalized = cdf2_r * float(hist2_r.max()/cdf2_r.max())
cdf2_g_normalized = cdf2_g * float(hist2_g.max()/cdf2_g.max())
cdf2_b_normalized = cdf2_b * float(hist2_b.max()/cdf2_b.max())
cdf2_normalized = [cdf2_r_normalized, cdf2_g_normalized, cdf2_b_normalized]

for i in range(3):
    plt.subplot(2,4,i+5)
    plt.plot(cdf2_normalized[i], color='black')
    plt.hist(img2[:,:,i].flatten(), 256, [0,256], color=hist_color[i])

plt.subplot(2,4,8)
plt.imshow(img2)

plt.show()

print(hist_b.size)

    