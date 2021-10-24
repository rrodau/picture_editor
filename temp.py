#!/sr/bin/env python3

import cv2
import numpy as np

BRIGHNTESS_CONTRAST_MIN_VALUE = 50
save_count = 1
# dummy function
def dummy(value):
    pass

# define convolution kernels
identity_kernel = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
])

sharpen_kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

gaussian_kernel1 = cv2.getGaussianKernel(3, 0)
gaussian_kernel2 = cv2.getGaussianKernel(7, 0)

box_kernel = np.ones(shape=(3, 3), dtype=np.float32) / 9.0
kernels = [identity_kernel, sharpen_kernel, gaussian_kernel1, gaussian_kernel2, box_kernel]

# read in an image, make a grayscale copy
original_img = cv2.imread('dog.jpg')
gray_original = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
w,h,c = original_img.shape
# create the UI (window and trackbars)
cv2.namedWindow('app')

# arguments: trackbarName, windowName, value(initial value), count(max value), onChange(event handler) 
cv2.createTrackbar('contrast', 'app', BRIGHNTESS_CONTRAST_MIN_VALUE, 100, dummy)
cv2.createTrackbar('brightness', 'app', BRIGHNTESS_CONTRAST_MIN_VALUE, 100, dummy)
cv2.createTrackbar('filter', 'app', 0, len(kernels)-1, dummy)
cv2.createTrackbar('grayscale', 'app', 0, 1, dummy)

def apply_changes(img, second_img, contrast, brightness):
    """
    a function that applys the brightness, contrast and filters to the given image

        @params: 
            img (numpy.ndarray): the images to which changes needs to be applied
            second_img (numpy.ndarray): a sparse matrix
            contrast (int): the value of the contrast
            brightness (int): the value of the brightness 

        @return: 
            a new image with contrast and brightness filter applied
    """
    img = cv2.filter2D(img, -1, kernels[filter_idx])
    return cv2.addWeighted(img, 1+(contrast-(BRIGHNTESS_CONTRAST_MIN_VALUE-1))*0.02, np.zeros_like(second_img), 0, brightness-BRIGHNTESS_CONTRAST_MIN_VALUE)

# main UI loop
while True:
    grayscale = cv2.getTrackbarPos('grayscale', 'app')
    contrast = cv2.getTrackbarPos('contrast', 'app')
    brightness = cv2.getTrackbarPos('brightness', 'app')
    filter_idx = cv2.getTrackbarPos('filter', 'app')
    # wait for keypress (100 milliseconds)
    key = cv2.waitKey(100)
    # quit
    if key == ord('q'):
        break
    # save image
    elif key == ord('s'):
        cv2.imwrite(f"output{1}.jpg", curr_img)
        save_count += 1
    # show image
    if grayscale == 0:
        second_img = np.zeros_like(original_img)
        curr_img = apply_changes(original_img, second_img, contrast, brightness)
    else:
        second_img = np.zeros_like(gray_original)
        curr_img = apply_changes(gray_original, second_img, contrast, brightness)
    cv2.imshow('app', curr_img)
# Window cleanup
cv2.destroyAllWindows()