from cv2 import BORDER_TRANSPARENT, WARP_INVERSE_MAP
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
MIN_MATCH_COUNT = 10
img1 = cv.imread('panorama/newnewpic1.jpg')          # queryImage
img2 = cv.imread('panorama/newnewpic2.jpg')
img3 = cv.imread('panorama/newnewpic3.jpg')
img1 = cv.resize(img1, None, fx=0.4, fy=0.4,interpolation=cv.INTER_LINEAR)
img2 = cv.resize(img2, None, fx=0.4, fy=0.4,interpolation=cv.INTER_LINEAR)
img3 = cv.resize(img3, None, fx=0.4, fy=0.4,interpolation=cv.INTER_LINEAR)
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)          # queryImage
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY) # trainImage
gray3 = cv.cvtColor(img3, cv.COLOR_BGR2GRAY) # trainImage
# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(gray1,None)    #(mask)특정 영역을 주면 성능이 향상된다
kp2, des2 = sift.detectAndCompute(gray2,None)
kp3, des3 = sift.detectAndCompute(gray3,None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)   #가까운 점 50개 점 search(단 이상적인 50개 가까운 점은 아니다)
flann = cv.FlannBasedMatcher(index_params, search_params)   #input parameter들이 dictionary, FLANN은 자기와 가장 비슷한 점을 찾는 알고리즘
'''
    flann : feature 점 매칭하는데 단순 euclidean distance가 아니라 좀 더 기하학적인 알고리즘
    hirechical clutering과 비슷하다. trees = 5면 5개 계층이 생기는 것
'''
matches1 = flann.knnMatch(des1,des2,k=2) #k=2 가장 가까운 점 2개를 찾는다.(2개 쓰는 이유는 ratio distance)
matches2 = flann.knnMatch(des2,des3,k=2)
#des1=gray1, des2=gray2, 각 점들 갯수가 다 다르다.
# store all the good matches as per Lowe's ratio test.

def matchmaker(match): 
    good = []
    for m,n in match:
        if m.distance < 0.7*n.distance:
            good.append(m)
    return good

good1 = matchmaker(matches1)
good2 = matchmaker(matches2)

h,w = gray1.shape
        
def matrix_maker(good, kp_first, kp_second):
    src_pts = np.float32([ kp_first[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp_second[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    #Ransac은 ratio distance를 만족하는 점(ex.100개) 중 일정 값만 추린다. (outlier는 제외시킴)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)    
    print(M, len(mask))
    matchesMask = mask.ravel().tolist()
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    return M

M1 = matrix_maker(good1, kp1, kp2)
M2 = matrix_maker(good2, kp2, kp3)
invert_M1 = np.linalg.inv(M1)
invert_M2 = np.linalg.inv(M2)

print(M2)

before_vertex = [[0,0,1],[0,h-1,1],[w-1,0,1],[w-1,h-1,1]]
before_vertex = np.array(before_vertex).transpose()

after_vertex1 = np.matmul(M1, before_vertex)
after_vertex1 = after_vertex1 / after_vertex1[2, :]
after_vertex1 = after_vertex1[:2, :]
after_vertex1 = np.round(after_vertex1, 0).astype(np.int)
after_vertex2 = np.matmul(invert_M2, before_vertex)
after_vertex2 = after_vertex2 / after_vertex2[2, :]
after_vertex2 = after_vertex2[:2, :]
after_vertex2 = np.round(after_vertex2, 0).astype(np.int)

print(after_vertex1)
print(after_vertex2)
after_vertex_concat = np.concatenate([after_vertex1, after_vertex2], axis=1)
print(after_vertex_concat)

max1_x, max1_y = max(after_vertex_concat[0]), max(after_vertex_concat[1])
min1_x, min1_y = min(after_vertex_concat[0]), min(after_vertex_concat[1])

max2_x, max2_y = max(after_vertex_concat[0]), max(after_vertex_concat[1])
min2_x, min2_y = min(after_vertex_concat[0]), min(after_vertex_concat[1])

min_frameX, min_frameY = min(min1_x, min2_x), min(min1_y, min2_y)
max_frameX, max_frameY = max(max1_x, max2_x), max(max1_y, max2_y)

print(min_frameX, min_frameY)
print(max_frameX, max_frameY)

img4 = np.zeros((max_frameY-min_frameY+100, max_frameX-min_frameX+100, 3), dtype=np.uint8)
print(img3.shape)

for y in range(h):
    for x in range(w):
        point = [[x, y, 1]]
        point = np.array(point).transpose()

        after_point = np.matmul(M1, point)
        after_point = after_point / after_point[2, :] # z축 1로 설정
        after_point = after_point[:2, :] # z축 삭제
        after_point = np.round(after_point, 0)
        img4[int(after_point[1]) - min_frameY, int(after_point[0]) - min_frameX, :] = img1[y,x, :]
        
for y in range(h):
    for x in range(w):
        point = [[x, y, 1]]
        point = np.array(point).transpose()
        
        after_point2 = np.matmul(invert_M2, point)
        after_point2 = after_point2 / after_point2[2, :]
        after_point2 = after_point2[:2, :]
        after_point2 = np.round(after_point2, 0)
        img4[int(after_point2[1]) - min_frameY, int(after_point2[0]) - min_frameX, :] = img3[y, x, :]
        
        
for y in range(h):
    for x in range(w):
        img4[y - min_frameY, x - min_frameX, :] = img2[y, x, :]

plt.imshow(cv.cvtColor(img4, cv.COLOR_BGR2RGB))
plt.show()

outer_x = w*6
outer_y = h*5
panorama1 = cv.warpPerspective(img1, M1, (2*w, h))
panorama2 = cv.warpPerspective(img3, M2, (2*w, h), flags=WARP_INVERSE_MAP)

print(panorama2.shape)

# 왼쪽 사진을 원근 변환한 왼쪽 영역에 합성

# for x in range(h):
#     for y in range(w):
#         if all(panorama[x, y, :]) == 0:
#             panorama[x,y,:] = img1[x,y,:]

panorama2[0:h, 0:w, :] = img2
panorama1_rgb = cv.cvtColor(panorama1, cv.COLOR_BGR2RGB)
panorama2_rgb = cv.cvtColor(panorama2, cv.COLOR_BGR2RGB)

plt.imshow(panorama1_rgb)
plt.show()
plt.imshow(panorama2_rgb)
plt.show()

# cv.imshow('original', img2)
# cv.imshow('perspective', panorama)
# cv.waitKey(0)

    

    
    
    
# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                    singlePointColor = None,
#                    matchesMask = matchesMask, # draw only inliers
#                    flags = 2)
# img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
# plt.imshow(img3, 'res'),plt.show()
