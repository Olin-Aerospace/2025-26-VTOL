"""
This contains a class with functions that handle image stitching for one pair of images at a time.
This is the final code I've put together; map_stitching_old.py is an older approach I used for
reference.
"""

import cv2
import numpy as np

# defining the stitcher as a class for usability
class Pair_Stitcher():
    """
    A class that uses a SIFT algorithm through OpenCV for feature matching and image stitching.
    Steps of the process are split into methods, but in most cases the top-level stitch() method
    should be enough.
    
    Attributes:
        ratio (float): A float between 0 and 1 representing the minumum confidence in a feature
        match needed for it to be counted as a "good match" and used in image stitching. A higher
        value will lead to fewer but more reliable matches, and a lower value will lead to more but
        less reliable matches.
        
        min_match (int): An integer representing the minimum number of matching points between two
        images for the program to go through with image stitching. Stitching will not be attempted
        if there are fewer than this many matches.
        
        smoothing_window_size (int): An integer representing the number of pixels in the horizontal
        center of the image in which the panorama will fade (smooth) between the two images.
        
        sift (cv2.xfeatures2d_SIFT): An xfeatures2d_SIFT object, defined by OpenCV (versions 3.4.2
        or earlier). This class is used for finding the key points and features in each image using
        a SIFT algorithm.
    """
    
    def __init__(self, ratio=0.15, min_match=10, smoothing_window=800):
        """
        Initializes the Pair_Stitcher class.
        
        Args:
            ratio (float, optional): A float between 0 and 1 representing the minumum confidence in
            a feature match needed for it to be counted as a "good match" and used in image
            stitching. A higher value will lead to fewer but more reliable matches, and a lower
            value will lead to more but less reliable matches. Defaults to 0.15 (15% confidence).
        
            min_match (int, optional): An integer representing the minimum number of matching
            points between two images for the program to go through with image stitching. Stitching
            will not be attempted if there are fewer than this many matches. Defaults to 10.
            
            smoothing_Window (int, optional): An integer representing the number of pixels in the
            horizontal center of the image in which the panorama will fade (smooth) between the two
            images. Defaults to 800.
        """
        self.ratio = ratio
        self.min_match = min_match
        self.smoothing_window_size = smoothing_window
        self.sift = cv2.xfeatures2d.SIFT_create()
        # if xfeatures2d or SIFT_create are giving errors, make sure you have opencv version 3.4.2
        # or earlier. Same for opencv-contrib-python. It doesn't work in newer versions.
    
    def stitch(self, img1, img2):
        """
        A top-level method that finds the homography matrix and makes a mask to blend and stitch
        two images together.

        Args:
            img1 (np.ndarray): A numpy array representing the first image to be stitched.
            
            img2 (np.ndarray): A numpy array representing the second image to be stitched.
        
        Returns:
            np.ndarray: A numpy matrix representing the panorama of the two stitched images.
        """
        # define the homography matrix and the dimensions of the final panorama
        H = self.define_homography(img1, img2)
        height = img1.shape[0]
        width = img1.shape[1] + img2.shape[1]
        panorama1 = np.zeros((height, width, 3))
        
        # make the mask for the first image and use it to modify the first image
        mask1 = self.create_mask(img1, img2, side='left')
        panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
        panorama1 *= mask1
        
        # Warp the second image to match the first image's perspective, and apply a mask
        mask2 = self.create_mask(img1, img2, side='right')
        panorama2 = cv2.warpPerspective(img2, H, (width, height))*mask2
        
        # combine the two halves and clear out blank space (while maintaining a rectangle shape)
        panorama_full = panorama1 + panorama2
        rows, cols = np.where(panorama_full[:,:,0] != 0)
        min_row, max_row = min(rows), max(rows)+1
        min_col, max_col = min(cols), max(cols)+1
        return panorama_full[min_row:max_row, min_col:max_col, :]

    def define_homography(self, img1, img2):
        """
        Creates and returns the homography matrix for the two images to be stitched.

        Args:
            img1 (np.ndarray): A numpy matrix representing the first image to be stitched.
                        
            img2 (np.ndarray): A numpy matrix representing the second image to be stitched.

        Returns:
            np.ndarray: A numpy matrix representing the homography matrix of the images to be
            stitched.
        """
        # detecting key points and comparing them across the two images to find the two best
        # matches from image 2 for each point in image 1 with brute force matcher's KNN algorithm
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        
        # filters out bad matches based on confidence ratio. If the best match is better than the
        # second-best match by a factor of the confidence ratio, declare it a good match.
        good_points = []
        good_matches = []
        for m1, m2 in raw_matches:
            if m1.distance < (1-self.ratio) * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])
        
        # throws an error if there aren't enough good matches
        if len(good_points) < self.min_match:
            raise RuntimeError("Not enough good matches between images for stitching")
            
        # create lists of matching key points in each image at equivalent indices
        image1_kp = np.float32(
            [kp1[i].pt for (_, i) in good_points])
        image2_kp = np.float32(
            [kp2[i].pt for (i, _) in good_points])
        
        # make and return the homography matrix for the matches
        H, _ = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC, 5.0)
        return H
    
    def create_mask(self, img1, img2, side):
        """
        Creates and returns a mask to be applied to the panorama for one of the two images, decided
        by the side parameter. The mask is the intensity of each color from one image to be carried
        over into the final panorama.

        Args:
            img1 (np.ndarray): A numpy matrix representing the first image to be stitched.
                        
            img2 (np.ndarray): A numpy matrix representing the second image to be stitched.
            
            side (string): A string containing either 'left' or 'right',  identifying which of the
            two images the mask is being made for.

        Returns:
            np.ndarray: A numpy matrix representing the mask that will fade the selected image into
            the other. It has identical r, g, and b components to smooth all colors together.
        """
        # make dimensions of final panorama image
        height = img1.shape[0]
        width = img1.shape[1] + img2.shape[1]
        offset = int(self.smoothing_window_size/2)
        barrier = img1.shape[1] - offset
        mask = np.zeros((height, width))
        
        # make mask that fades from one image to the other in the middle, in an area of a width
        # defined by the smoothing_window
        if side == 'left':
            mask[:, barrier-offset:barrier+offset] = np.tile(np.linspace(1, 0, 2*offset).T, (height, 1))
            mask[:, :barrier-offset] = 1
        else:
            mask[:, barrier-offset:barrier+offset] = np.tile(np.linspace(0, 1, 2*offset).T, (height, 1))
            mask[:, barrier+offset:] = 1
        
        # returns a 3-layer mask to apply to r, g, and b components of image
        return cv2.merge([mask, mask, mask])