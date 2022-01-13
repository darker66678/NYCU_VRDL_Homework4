import cv2
import os
import glob
import numpy as np


def gaussian_noise(img, mean=0, sigma=0.05):
    img = img / 255.0
    noise = np.random.normal(mean, sigma, img.shape)
    gaussian_out = img + noise
    gaussian_out = np.clip(gaussian_out, 0, 1)
    gaussian_out = np.uint8(gaussian_out*255)

    return gaussian_out


hr_folder = './data/train/hr/'
scale = 3
hr_imgs = glob.glob(os.path.join(hr_folder, "*.png"))
files = os.listdir(hr_folder)
divide = 3
image_num = 0
val = 0.05 * len(hr_imgs)
for index, hr_img in enumerate(hr_imgs):
    img = cv2.imread(hr_img, 1)
    divided_y = img.shape[0]//divide
    divided_x = img.shape[1]//divide
    # if(image_num > val):
    for i in range(divide):
        for j in range(divide):
            divided_img = img[divided_y*i:divided_y *
                              (i+1), divided_x*j:divided_x*(j+1)]
            cv2.imwrite(
                f'./data/train/divide_hr/{image_num}.png', divided_img)
            image_num += 1
            print(image_num)

            '''lr_y = divided_img.shape[0]//3
            lr_x = divided_img.shape[1]//3

            small_divided_img = cv2.resize(
                divided_img, (lr_x, lr_y), interpolation=cv2.INTER_AREA)
            cv2.imwrite(
                f'./data/train/divide_lr/{image_num}.png', small_divided_img)'''

    '''else:
    for i in range(divide):
        for j in range(divide):
            divided_img = img[divided_y*i:divided_y *
                              (i+1), divided_x*j:divided_x*(j+1)]
            cv2.imwrite(
                f'./data/val/divide_hr/{image_num}.png', divided_img)

            lr_y = divided_img.shape[0]//3
            lr_x = divided_img.shape[1]//3

            small_divided_img = cv2.resize(
                divided_img, (lr_x, lr_y), interpolation=cv2.INTER_AREA)
            cv2.imwrite(
                f'./data/val/divide_lr/{image_num}.png', small_divided_img)

            image_num += 1
            print(image_num)
    lr_y = img.shape[0]//3
    lr_x = img.shape[1]//3
    image = cv2.resize(img, (lr_x, lr_y), interpolation=cv2.INTER_AREA)
    # image = gaussian_noise(image)
    cv2.imwrite(f'./data/val/new_lr/{files[index]}', image)'''
