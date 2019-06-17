import cv2
from PIL import Image #Image from PIL
import numpy as np
import os


# directory = '/Users/byanbansal/Desktop/HotOrBot/new0/'
initialcount = 1
cropcount = 1

# resized_image = cv2.resize(cv2.imread(directory + 'cropped1.jpg'), (512, 512))
# cv2.imshow('img', resized_image)
# cv2.waitKey(0)


def faceDetectCrop(image):
    
    global cropcount
    
    face_cascade = cv2.CascadeClassifier('/Users/byanbansal/opencv/data/haarcascades/haarcascade_frontalface_alt.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.6, 3)

    if len(faces) > 0:
        for (x,y,w,h) in faces:
            print('Cropping ' + str(initialcount) + ' and saving to ' + str(cropcount))
            crop_img = image[y:y+h, x:x+w]
            return crop_img
            # cv2.imwrite(directory + 'newcrop' + str(cropcount) + '.jpg', crop_img)
            # cropcount = cropcount + 1

    else:
        print('No faces found in.' + str(initialcount))
        return None

''' for img in os.listdir(directory):
    
    image = cv2.imread(directory + img)
    
    if image is None:
        print('Image ' + str(initialcount) + ' could not be opened.')
        continue
    
    # os.rename(directory + img, directory + 'crop' + str(initialcount) + '.jpg')
    
    faceDetectCrop(image)
    initialcount = initialcount + 1 '''
