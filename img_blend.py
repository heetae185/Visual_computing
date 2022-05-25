import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img_eye = cv.imread('img_blending/stitch_eye.jpg')
img_hand = cv.imread('img_blending/stitch_hand.jpg')

def plot_img(index, img, title, plot_grid_size):
    plt.subplot(plot_grid_size[0],plot_grid_size[1],index)
    plt.imshow(img[...,::-1])
    plt.axis('off'), plt.title(title)

def display_untilKey(Pimgs, Titles, file_out = False):
    for img, title in zip(Pimgs, Titles):
        cv.imshow(title, img)
        if file_out == True:
            cv.imwrite(title + ".jpg", img)
    cv.waitKey(0)

def __pyrUp(img, size = None):
    nt = tuple([x*2 for x in (img.shape[1], img.shape[0])])
    if size == None:
        size = nt
    # bug?!
    if nt != size:
        upscale_img = cv.pyrUp(img)
        upscale_img = cv.resize(upscale_img, size)  #img 사이즈가 안 맞으면 error, 따라서 size 조정
    else:
        upscale_img = cv.pyrUp(img)
    return upscale_img

def generate_gaussian_pyramid(img, levels):
    GP = [img]
    for i in range(1, levels): # 1 to levels - 1 same as range(1, levels, 1)
        img = cv.pyrDown(img)
        GP.append(img)
    return GP

def generate_laplacian_pyramid(GP):
    levels = len(GP)
    LP = [] #[GP[levels + 1]]
    for i in range(levels - 1, 0, -1):
        upsample_img = __pyrUp(GP[i], (GP[i-1].shape[1], GP[i-1].shape[0]))
        laplacian_img = cv.subtract(GP[i-1], upsample_img)
        LP.append(laplacian_img)
    LP.reverse()
    return LP

def generate_pyramid_composition_image(Pimgs):
    levels = len(Pimgs)
    #print(levels)
    rows, cols = (Pimgs[0].shape[0], Pimgs[0].shape[1])
    composite_image = np.zeros((rows+levels, cols + int(cols / 2 + 0.5), 3), dtype=Pimgs[0].dtype)
    composite_image[:rows, :cols, :] = Pimgs[0].reshape(rows, cols, 3)  #image size 조정
    i_row = 0
    for p in Pimgs[1:]:
        n_rows, n_cols = p.shape[0], p.shape[1]
        composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
        i_row += n_rows
    return composite_image

#이미지 삽입 함수
def stitch_inside(P_out, P_in):
    P_stitch = []
    for la,lb in zip(P_out, P_in):
        in_col = lb.shape[1]
        in_row = lb.shape[0]
        for i in range(3):
            la[la.shape[0]//2-in_row//2:la.shape[0]//2-in_row//2+in_row, \
               la.shape[1]//2-in_col//2:la.shape[1]//2-in_col//2+in_col, i] = lb[:,:,i]
        P_stitch.append(la)
    return P_stitch

#just for check
print(img_hand.shape, img_eye.shape)

#몇 level에서 blending을 시작할지에 따라 blending 결과물을 return 하는 함수 
def stitch_level(level):
    
    plot_grid_size = (3, level)
    
    GP_hand = generate_gaussian_pyramid(img_hand, level)
    GP_eye = generate_gaussian_pyramid(img_eye, level)
    LP_hand = generate_laplacian_pyramid(GP_hand)
    LP_eye = generate_laplacian_pyramid(GP_eye)
    
    '''
    if level > 2:
        display_untilKey([generate_pyramid_composition_image(GP_hand), 
                    generate_pyramid_composition_image(GP_eye),
                    generate_pyramid_composition_image(LP_hand),
                    generate_pyramid_composition_image(LP_eye)], 
                    ["GP_hand", "GP_eye", "LP_hand", "LP_eye"])'''

    
    LP_stitch = stitch_inside(LP_hand, LP_eye)

    img_stitch = stitch_inside([img_hand], [img_eye])[0]
    GP_stitch = generate_gaussian_pyramid(img_stitch, level)
    
    '''
    if level > 2 :
        display_untilKey([generate_pyramid_composition_image(GP_stitch),
                    generate_pyramid_composition_image(LP_stitch)], 
                    ["composite GP imgs", "composite LP imgs"])'''

    recon_img = GP_stitch[-1] 
    lp_maxlev = len(LP_stitch) - 1
    #plot_img(6, recon_img.copy(), "level: " + str(level), plot_grid_size)
    print(lp_maxlev)
    for i in range(lp_maxlev, -1, -1):
        recon_img = __pyrUp(recon_img, (LP_stitch[i].shape[1], LP_stitch[i].shape[0]))
        #plot_img(i + 1 + level*2, recon_img.copy(), "level: " + str(i), plot_grid_size)
        recon_img = cv.add(recon_img, LP_stitch[i])
        #plot_img(i + 1, recon_img.copy(), "level: " + str(i), plot_grid_size)
        #plot_img(i + 1 + level, LP_stitch[i].copy(), "level: " + str(i), plot_grid_size)
        
    recon_img = cv.cvtColor(recon_img, cv.COLOR_BGR2RGB)
    return recon_img
    
recon_img = []
for i in range(8):
    recon_img.append(stitch_level(i))

plt.figure(figsize=(12,6))
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(recon_img[i])
    plt.title('Level {}'.format(i))
plt.show()

#high level 이미지에 sharpen, medium level 이미지에 blur 필터 입혀보기
kernel = np.array(np.mat('0,-1,0;-1,5,-1;0,-1,0'))  
img_sharp = cv.filter2D(recon_img[7], -1, kernel)
img_blur = cv.blur(recon_img[5],(5,5))
img_gaussianblur = cv.GaussianBlur(recon_img[5],(5,5),0)
    
#level7 sharpen 적용
plt.subplot(1,2,1)
plt.imshow(recon_img[7])
plt.title('level7 original')
plt.subplot(1,2,2)
plt.imshow(img_sharp)
plt.title('level7 sharpen')
plt.show()

#level5 gaussian blur 적용
plt.figure(figsize=(10,4))
plt.subplot(1,3,1)
plt.imshow(recon_img[5])
plt.title('level5 original')
plt.subplot(1,3,2)
plt.imshow(img_blur)
plt.title('level5 blur')
plt.subplot(1,3,3)
plt.imshow(img_gaussianblur)
plt.title('level5 gaussianblur')
plt.show()

#최종 결과 : level5, 6에 gaussian blur 적용
level5_gaussianblur = cv.blur(recon_img[5],(5,5))
level6_gaussianblur = cv.blur(recon_img[6],(5,5))
plt.subplot(1,2,1)
plt.imshow(level5_gaussianblur)
plt.title('level5 gaussianblur')
plt.subplot(1,2,2)
plt.imshow(level6_gaussianblur)
plt.title('level6 gaussianblur')
plt.show()
