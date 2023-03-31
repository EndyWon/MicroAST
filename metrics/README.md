## The codes for calculating SSIM score and Style Loss/Content Loss.

The files should be organized with the following name format:
   
- content images: `x.jpg` (e.g., `1.jpg`)

- style images: `y.jpg` (e.g., `2.jpg`)

- stylized images: `x_stylized_y.jpg` (e.g., `1_stylized_2.jpg`)


### Usage

- Calculating the SSIM scores between stylized images and content images:

`python calc_ssim.py --content_dir DIR1 --stylized_dir DIR2`

- Calculating the Style Loss between stylized images and style images:

`python calc_cs_loss.py --style_dir DIR1 --stylized_dir DIR2 --mode 0`

- Calculating the Content Loss between stylized images and content images:

`python calc_cs_loss.py --content_dir DIR1 --stylized_dir DIR2 --mode 1`

- Calculating both Style Loss and Content Loss: 

`python calc_cs_loss.py --content_dir DIR1 --style_dir DIR2 --stylized_dir DIR3 --mode 2`
