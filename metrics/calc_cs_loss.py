import argparse
import os
import torch
import torch.nn as nn

import cv2


# construct the argument parse and parse the arguments
"""IMPORTANT Note: Files' name format:
   content images: x.jpg (e.g., 1.jpg)
   style images: y.jpg (e.g., 2.jpg)
   stylized images: x_stylized_y.jpg (e.g., 1_stylized_2.jpg) """

parser = argparse.ArgumentParser()
parser.add_argument("--content_dir", help="the directory of content images")
parser.add_argument("--style_dir", help="the directory of style images")
parser.add_argument("--stylized_dir", required=True, help="the directory of stylized images")
parser.add_argument('--mode', type=int, default=0, help="0 for style loss, 1 for content loss, 2 for both")
args = parser.parse_args()


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(inplace=True),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(inplace=True),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(inplace=True),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(inplace=True),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(inplace=True),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(inplace=True),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(inplace=True),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(inplace=True),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(inplace=True),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(inplace=True),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(inplace=True),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(inplace=True),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(inplace=True),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(inplace=True),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(inplace=True),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(inplace=True)  # relu5-4
)

vgg.eval()
vgg.load_state_dict(torch.load("../models/vgg_normalised.pth"))


enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1

enc_1.to(device)
enc_2.to(device)
enc_3.to(device)
enc_4.to(device)
enc_5.to(device)


####################################################################################################################
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std



def calc_content_loss(input, target):
    assert (input.size() == target.size())
    return torch.nn.MSELoss()(input, target)

def calc_style_loss(input, target):
    #assert (input.size() == target.size())
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)
    return torch.nn.MSELoss()(input_mean, target_mean) + torch.nn.MSELoss()(input_std, target_std)


content_dir = args.content_dir
style_dir = args.style_dir
stylized_dir = args.stylized_dir

stylized_files = os.listdir(stylized_dir)

with torch.no_grad():
    if args.mode == 0 or args.mode == 2:
        loss_s_sum  = 0.
        count = 0

        for stylized in stylized_files:
            stylized_img = cv2.imread(stylized_dir + stylized)   # stylized image
            # stylized_img = cv2.resize(stylized_img, (256,256))

            name = os.path.splitext(stylized)[0].split("_")  # parse the style image's name

            style_img = cv2.imread(style_dir + name[2] + '.jpg')  # style image
            # style_img = cv2.resize(style_img, (256,256))

            stylized_img = torch.tensor(stylized_img, dtype=torch.float)
            stylized_img = stylized_img/255
            stylized_img = torch.unsqueeze(stylized_img, dim=0)
            stylized_img = stylized_img.permute([0, 3, 1, 2])
            stylized_img = stylized_img.cuda().to(device)

            style_img = torch.tensor(style_img, dtype=torch.float)
            style_img = style_img/255
            style_img = torch.unsqueeze(style_img, dim=0)
            style_img = style_img.permute([0, 3, 1, 2])
            style_img = style_img.cuda().to(device)


            loss_s = 0.

            o1_1 = enc_1(stylized_img)
            s1_1 = enc_1(style_img)
            loss_s += calc_style_loss(o1_1,s1_1)

            o2_1 = enc_2(o1_1)
            s2_1 = enc_2(s1_1)
            loss_s += calc_style_loss(o2_1,s2_1)


            o3_1 = enc_3(o2_1)
            s3_1 = enc_3(s2_1)
            loss_s += calc_style_loss(o3_1,s3_1)


            o4_1 = enc_4(o3_1)
            s4_1 = enc_4(s3_1)
            loss_s += calc_style_loss(o4_1,s4_1)

            o5_1 = enc_5(o4_1)
            s5_1 = enc_5(s4_1)
            loss_s += calc_style_loss(o5_1,s5_1)

            print ("Style Loss: {}".format(loss_s/5))
            loss_s_sum += float(loss_s/5)
            count += 1


        print ("Total num: {}".format(count))
        print ("Average Style Loss: {}".format(loss_s_sum/count))


    if args.mode == 1 or args.mode == 2:
        loss_c_sum  = 0.
        count = 0

        for stylized in stylized_files:
            stylized_img = cv2.imread(stylized_dir + stylized)   # stylized image
            # stylized_img = cv2.resize(stylized_img, (256,256))

            name = stylized.split("_")  # parse the content image's name

            content_img = cv2.imread(content_dir + name[0] + '.jpg')   # content image
            # content_img = cv2.resize(content_img, (256,256))

            stylized_img = torch.tensor(stylized_img, dtype=torch.float)
            stylized_img = stylized_img/255
            stylized_img = torch.unsqueeze(stylized_img, dim=0)
            stylized_img = stylized_img.permute([0, 3, 1, 2])
            stylized_img = stylized_img.cuda().to(device)

            content_img = torch.tensor(content_img, dtype=torch.float)
            content_img = content_img/255
            content_img = torch.unsqueeze(content_img, dim=0)
            content_img = content_img.permute([0, 3, 1, 2])
            content_img = content_img.cuda().to(device)

            loss_c = 0.

            o1 = enc_4(enc_3(enc_2(enc_1(stylized_img))))
            c1 = enc_4(enc_3(enc_2(enc_1(content_img))))

            loss_c += calc_content_loss(o1,c1)

            o2 = enc_5(o1)
            c2 = enc_5(c1)
            loss_c += calc_content_loss(o2,c2)


            print ("Content Loss: {}".format(loss_c/2))
            loss_c_sum += float(loss_c/2)
            count += 1


        print ("Total num: {}".format(count))
        print ("Average Content Loss: {}".format(loss_c_sum/count))



