#!/sr/bin/env python3

import cv2
import numpy as np

BRIGHNTESS_CONTRAST_MIN_VALUE = 50
# dummy function
def dummy(value):
    pass

# read in an image, make a grayscale copy
original_img = cv2.imread('dog.jpg')
gray_original = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
w,h,c = original_img.shape
# create the UI (window and trackbars)
cv2.namedWindow('app')

# arguments: trackbarName, windowName, value(initial value), count(max value), onChange(event handler) 
cv2.createTrackbar('contrast', 'app', BRIGHNTESS_CONTRAST_MIN_VALUE, 100, dummy)
cv2.createTrackbar('brightness', 'app', BRIGHNTESS_CONTRAST_MIN_VALUE, 100, dummy)
cv2.createTrackbar('filter', 'app', 0, 1, dummy) # TODO: update max value with number of filters
cv2.createTrackbar('grayscale', 'app', 0, 1, dummy)

def apply_brightness_and_contrast(img, second_img, contrast, brightness):
    """
    a function that applys the brightness and contrast filters to the given image

        @params: 
            img (numpy.ndarray): the images to which changes needs to be applied
            second_img (numpy.ndarray): a sparse matrix
            contrast (int): the value of the contrast
            brightness (int): the value of the brightness 

        @return: 
            a new image with contrast and brightness filter applied
    """
    return cv2.addWeighted(img, 1+(contrast-(BRIGHNTESS_CONTRAST_MIN_VALUE-1))*0.02, np.zeros_like(second_img), 0, brightness-BRIGHNTESS_CONTRAST_MIN_VALUE)

# main UI loop
while True:
    # TODO: read all of the trackbar values
    grayscale = cv2.getTrackbarPos('grayscale', 'app')
    contrast = cv2.getTrackbarPos('contrast', 'app')
    brightness = cv2.getTrackbarPos('brightness', 'app')
    # TODO: apply the filters
    # wait for keypress (100 milliseconds)
    key = cv2.waitKey(100)
    if key == ord('q'):
        break
    elif key == ord('s'):
        # TODO: save Image
        pass
    
    # show image
    if grayscale == 0:
        # TODO: replace with modified image
        second_img = np.zeros_like(original_img)
        curr_img = apply_brightness_and_contrast(original_img, second_img, contrast, brightness)
    else:
        second_img = np.zeros_like(gray_original)
        curr_img = apply_brightness_and_contrast(gray_original, second_img, contrast, brightness)
    cv2.imshow('app', curr_img)
# TODO: remove this line!
cv2.waitKey(0)
# Window cleanup
cv2.destroyAllWindows()