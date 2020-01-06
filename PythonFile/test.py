import utils
import cv2

img = cv2.imread('4.png', cv2.IMREAD_COLOR)
blue = utils.get_chars(img.copy(), utils.BLUE)
green = utils.get_chars(img.copy(), utils.GREEN)
red = utils.get_chars(img.copy(), utils.RED)

cv2.imshow('Image Gray', blue)
cv2.waitKey(0)
cv2.imshow('Image Gray', green)
cv2.waitKey(0)
cv2.imshow('Image Gray', red)
cv2.waitKey(0)




