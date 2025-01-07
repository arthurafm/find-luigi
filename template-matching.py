import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

imgs_rgb = []
imgs_greyscale = []
for file in [f'test{i}.png' for i in range(6)]:
    img = cv.imread(file, cv.COLOR_RGB2BGR)
    imgs_rgb.append(img)
    imgs_greyscale.append(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
imgs2 = imgs_greyscale.copy()
template = cv.imread('icon.png', cv.IMREAD_GRAYSCALE)
template = cv.resize(template, (70, 89))
w, h = template.shape[::-1]

methods = ['TM_CCOEFF_NORMED', 'TM_CCORR_NORMED', 'TM_SQDIFF_NORMED']

for i in range(6):
    print(f'Image nº{i+1}')
    for meth in methods:
        img = imgs2[i].copy()
        method = getattr(cv, meth)
    
        # Apply template Matching
        res = cv.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
            confidence = 1 - min_val
        else:
            top_left = max_loc
            confidence = max_val
        bottom_right = (top_left[0] + w, top_left[1] + h)
        
        print(f'Confidence: {confidence}')
        img_rgb = imgs_rgb[i].copy()
        cv.rectangle(img_rgb, top_left, bottom_right, (0, 0, 255), 2)

        plt.imshow(cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB))
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)
    
        plt.show()