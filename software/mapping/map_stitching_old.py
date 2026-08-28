# THIS IS THE OLD VERSION. RUN AT YOUR OWN RISK

import cv2
import numpy as np
import os
from PIL import Image

def warp_img(base, newImg):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(newImg, cv2.COLOR_BGR2GRAY)

    # Initialize the feature detector and extractor (e.g., SIFT)
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for both images
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Initialize the feature matcher using brute-force matching
    bf = cv2.BFMatcher()

    # Match the descriptors using brute-force matching
    matches = bf.match(descriptors1, descriptors2)

    # Select the top N matches
    num_matches = 50
    matches = sorted(matches, key=lambda x: x.distance)[:num_matches]

    # Extract matching keypoints
    src_points = np.float32([keypoints1[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    dst_points = np.float32([keypoints2[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

    # Estimate the homography matrix
    homography, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

    # Warp the first image using the homography
    result = cv2.warpPerspective(base, homography, (newImg.shape[1], newImg.shape[0]))

    return result

    # Blending the warped image with the second image using alpha blending
    #alpha = 0.5  # blending factor
    #blended_image = cv2.addWeighted(result, alpha, newImg, 1 - alpha, 0)

    
    # Save the blended image
    #cv2.imwrite('./blended_v2_result.jpg', result)
    #cv2.imwrite('./blended_v2.jpg', blended_image)

def remove_background(img):
    tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(img)
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)
    return dst

# START OF MAIN
dir = '/home/bmeyer/Downloads/stitch-test-pics/'
imgs = os.listdir(dir)
imgs.sort()
test_imgs = imgs[48:55] #50-52
#for name in os.listdir(dir):
#    imgs.append(cv2.imread(os.path.join(dir, name)))

for j in range(len(imgs)//2):
    test_imgs = imgs[j*2:(j+1)*2]
    for i in range(len(test_imgs)):
        if i == 0:
            img = cv2.imread(dir+test_imgs[i])
            img = cv2.copyMakeBorder(img, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=[0,0,0])
            img = remove_background(img)
            cv2.imwrite('./imgs/blended_v2_'+str(j)+'.png', img)
            continue

        # Load the images
        oldImg = cv2.imread(dir+test_imgs[i])
        newImg = cv2.imread('./imgs/blended_v2_'+str(j)+'.png')
        
        #add transparent border to image
        newImg = cv2.copyMakeBorder(newImg, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=[0,0,0])
        newImg = remove_background(newImg)

        #warp new image to fit base image
        warped_img = warp_img(oldImg, newImg)
        warped_transparent = remove_background(warped_img)

        #save images to computer
        cv2.imwrite('./blended_v2_base.png', newImg)
        cv2.imwrite('./blended_v2_result.png', warped_transparent)

        #overlay new image onto base
        base = Image.open('./blended_v2_base.png')
        new = Image.open('./blended_v2_result.png')
        new.paste(base, (0,0), mask=base)
        new.save('./imgs/blended_v2_'+str(j)+'.png', format="PNG")