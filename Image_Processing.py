import cv2
import numpy as np
import os

alpha = 1.5 # Multiplicative factor for intensity scaling
beta = 30 # Additive factor for intensity scaling
folder_path = '../Data/images/'
folder_result = "../Data/Imag_processing/"

if not os.path.exists("../Data/Imag_processing/"):
  os.makedirs("../Data/Imag_processing/")
  
def resize_image(image, size=(224,224)):
    resized_image = cv2.resize(image, size)
    return resized_image

def adjust_brightness(image, alpha, beta):
    new_image = np.zeros(image.shape, image.dtype)
    new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return new_image

def geometric_transform(image, transformation_matrix, dsize):
    transformed_image = cv2.warpAffine(image, transformation_matrix, dsize)
    return transformed_image

for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image = cv2.imread(os.path.join(folder_path, filename))
        if image is not None:
            image_resize  = resize_image(image)
            bright_image = adjust_brightness(image_resize, alpha, beta)
            rows, cols = bright_image.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1) 
            transformed_image = geometric_transform(image, rotation_matrix, (cols, rows))
            cv2.imwrite(os.path.join(folder_result, filename), transformed_image)
