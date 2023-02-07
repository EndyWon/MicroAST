from skimage.metrics import structural_similarity
import argparse

import cv2
import os

# construct the argument parse and parse the arguments
"""IMPORTANT Note: Files' name format:
   content images: x.jpg (e.g., 1.jpg)
   style images: y.jpg (e.g., 2.jpg)
   stylized images: x_y.jpg (e.g., 1_2.jpg) """

parser = argparse.ArgumentParser()
parser.add_argument("--content_dir", required=True, help="the directory of content images")
parser.add_argument("--stylized_dir", required=True, help="the directory of stylized images")
args = parser.parse_args()


content_dir = args.content_dir
stylized_dir = args.stylized_dir

stylized_files = os.listdir(stylized_dir)

ssim_sum = 0.
count = 0

for stylized in stylized_files:
    stylized_img = cv2.imread(stylized_dir + stylized)  # stylized image

    name = stylized.split("_")  # parse the content image's name
    content_img = cv2.imread(content_dir + name[0] + '.jpg')   # content image

    grayA = cv2.cvtColor(content_img, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(stylized_img, cv2.COLOR_BGR2GRAY)
    (score, diff) = structural_similarity(grayA, grayB, full=True)

    print("SSIM: {}".format(score))
    ssim_sum += score
    count += 1

print ("Total num: {}".format(count))
print("Average SSIM: {}".format(ssim_sum/count))





