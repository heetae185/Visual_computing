import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('color_histogram_equal/low_contrast5_4.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

IMG_ROW = img.shape[0]   #이미지 pixel 가로값
IMG_COL = img.shape[1]   #이미지 pixel 세로값

img2 = np.zeros((IMG_ROW, IMG_COL, 3))  #이미지 배열 미리 지정

def cdf_m_maker(cdf):
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    return cdf

hist_color=['r', 'g', 'b']
row_size = 30   #tilesize 60x60
col_size = 30   #tilesize 60x60

#original 버전 cdf그래프를 그리기 위한 과정
hist0_r,bins0_r = np.histogram(img[:,:,0].flatten(),256,[0,256])
hist0_g,bins0_g = np.histogram(img[:,:,1].flatten(),256,[0,256])
hist0_b,bins0_b = np.histogram(img[:,:,2].flatten(),256,[0,256])
cdf0_r = hist0_r.cumsum()
cdf0_g = hist0_g.cumsum()
cdf0_b = hist0_b.cumsum()
cdf0_r_normalized = cdf0_r * float(hist0_r.max()/cdf0_r.max())
cdf0_g_normalized = cdf0_g * float(hist0_g.max()/cdf0_g.max())
cdf0_b_normalized = cdf0_b * float(hist0_b.max()/cdf0_b.max())
cdf0_normalized = [cdf0_r_normalized, cdf0_g_normalized, cdf0_b_normalized]

#패딩 하지 않고 모서리를 8분할 하여 다르게 적용
def img_slicer(img, row, col):
    if (row < row_size) :
        if (col_size < col < IMG_COL-col_size): #상단 중앙
            img = img[0:row+row_size, col-col_size:col+col_size]
        elif (col_size > col):  #상단 왼쪽
            img = img[0:row+row_size, 0:col+col_size]
        elif (col > IMG_COL-col_size):  #상단 오른쪽
            img = img[0:row+row_size, col-col_size:IMG_COL]
    elif (row > IMG_ROW-row_size):
        if (col_size < col < IMG_COL-col_size): #하단 중앙
            img = img[row-row_size:IMG_ROW, col-col_size:col+col_size]
        elif (col_size > col):  #하단 왼쪽
            img = img[row-row_size:IMG_ROW, 0:col+col_size]
        elif (col > IMG_COL-col_size):  #하단 오른쪽
            img = img[row-row_size:IMG_ROW, col-col_size:IMG_COL]
    #왼쪽 모서리
    elif ((row_size < row < IMG_ROW-row_size) and (col < col_size)):
        img = img[row-row_size:row+row_size, 0:col+col_size]
    #오른쪽 모서리
    elif ((row_size < row < IMG_ROW-row_size) and (col > IMG_COL-col_size)):
        img = img[row-row_size:row+row_size, col-col_size:IMG_COL]
    #가운데
    else:
        img = img[row-row_size:row+row_size, col-col_size:col+col_size]
    return img
                

#픽셀 별로 다른 히스토그램 적용(adaptive), 또한 패딩 하지 않고 모서리를 8분할 하여 다르게 적용
for row in range(IMG_ROW):
    for col in range(IMG_COL):
        hist_r,bins_r = np.histogram(img_slicer(img,row,col).flatten(),256,[0,256])
        hist_g,bins_g = np.histogram(img_slicer(img,row,col).flatten(),256,[0,256])
        hist_b,bins_b = np.histogram(img_slicer(img,row,col).flatten(),256,[0,256])
        
        cdf_r = hist_r.cumsum()
        cdf_g = hist_g.cumsum()
        cdf_b = hist_b.cumsum()
        cdf_r_normalized = cdf_r * float(hist_r.max()/cdf_r.max())
        cdf_g_normalized = cdf_g * float(hist_g.max()/cdf_g.max())
        cdf_b_normalized = cdf_b * float(hist_b.max()/cdf_b.max())
        cdf_normalized = [cdf_r_normalized, cdf_g_normalized, cdf_b_normalized]

        cdf_r = cdf_m_maker(cdf_r)
        cdf_g = cdf_m_maker(cdf_g)
        cdf_b = cdf_m_maker(cdf_b)

        #각 픽셀에 대한 histogram을 바탕으로 픽셀 입력
        img2[row,col,0] = cdf_r[img[row,col,0]]
        img2[row,col,1] = cdf_g[img[row,col,1]]
        img2[row,col,2] = cdf_b[img[row,col,2]]
        
img2 = img2.astype('uint8')

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
        
plt.figure(figsize=(12,8))
for i in range(3):
    plt.subplot(2,4,i+1)
    plt.plot(cdf0_normalized[i], color='black')
    plt.hist(img[:,:,i].flatten(), 256, [0,256], color=hist_color[i])
    plt.xlim([0,256])
plt.subplot(2,4,4)
plt.imshow(img)

for i in range(3):
    plt.subplot(2,4,i+5)
    plt.plot(cdf2_normalized[i], color='black')
    plt.hist(img2[:,:,i].flatten(), 256, [0,256], color=hist_color[i])
plt.subplot(2,4,8)
plt.imshow(img2)

plt.show()
