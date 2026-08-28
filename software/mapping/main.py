from pair_stitcher import Pair_Stitcher
import cv2
import os

num_images = 6 # CHANGE THIS!! MESS WITH IT IF YOU WANT!!!!!

# instantialize a stitcher. ratio and min_match should be fine, but you'll have to mess with
# smoothing_window for different-sized images. See pair_stitcher.py for more detail.
stitcher = Pair_Stitcher(ratio=0.15, min_match=10, smoothing_window=500)

# define directory with images to stitch and make a list of the images' filenames
dir = 'test_images/'
imgs = os.listdir(dir)
imgs.sort()

# loop through images in the directory to stitch them together. This saves the stitched image to
# this directory, then loads it back and stitches it onto the next image. Two images at a time,
# repeated until every image has been added to the panorama. You can watch it in real time by
# keeping panorama.jpg open while the code runs, since the file is overwritten every iteration.
old_img = cv2.imread(dir + imgs[0])
for i in range(num_images-1):
    new_img = cv2.imread(dir + imgs[i+1])
    pano = stitcher.stitch(new_img, old_img)
    cv2.imwrite('panorama.jpg', pano)
    old_img = cv2.imread('panorama.jpg')