# MicroAST（AAAI 2023）
**[update 8/28/2022]**

Official Pytorch code for ["MicroAST: Towards Super-Fast Ultra-Resolution Arbitrary Style Transfer"](https://arxiv.org/pdf/2211.15313.pdf)

## Introduction:

**MicroAST** is a lightweight model that completely abandons the use of cumbersome pre-trained Deep Convolutional Neural Networks (e.g., VGG) at inference. Instead, two micro encoders (content and style encoders) and one micro decoder are utilized for style transfer. The content encoder aims at extracting the main structure of the content image. The style encoder, coupled with a modulator, encodes the style image into learnable dual-modulation signals that modulate both intermediate features and convolutional filters of the decoder, thus injecting more sophisticated and flexible style signals to guide the stylizations. In addition, to boost the ability of the style encoder to extract more distinct and representative style signals, it also introduces a new style signal contrastive loss. MicroAST is 5-73 times smaller and 6-18 times faster than the state of the art, for the first time enabling super-fast (about 0.5 seconds) arbitrary style transfer at 4K ultra-resolutions. 

![show](https://github.com/EndyWon/MicroAST/blob/main/figures/teaser.jpg)

## Environment:
- Python 3.6
- Pytorch 1.8.0

## Getting Started:
**Clone this repo:**

`git clone https://github.com/EndyWon/MicroAST`  
`cd MicroAST`

**Test:**

- Test a pair of images:

  `python test_microAST.py --content inputs/content/1.jpg --style inputs/style/1.jpg`
  
- Test two collections of images:

  `python test_microAST.py --content_dir inputs/content/ --style_dir inputs/style/`

**Train:**

- Download content dataset [MS-COCO](https://cocodataset.org/#download) and style dataset [WikiArt](https://www.kaggle.com/c/painter-by-numbers) and then extract them.

- Download the pre-trained [vgg_normalised.pth](https://drive.google.com/file/d/1PUXro9eqHpPs_JwmVe47xY692N3-G9MD/view?usp=sharing), place it at path `models/`.

- Run train script:

  `python train_microAST.py --content_dir ./coco2014/train2014 --style_dir ./wikiart/train`
  
  
 ## Citation:

If you find the ideas and codes useful for your research, please cite the paper:

```
@inproceedings{wang2023microast,
  title={MicroAST: Towards Super-Fast Ultra-Resolution Arbitrary Style Transfer},
  author={Wang, Zhizhong and Zhao, Lei and Zuo, Zhiwen and Li, Ailin and Chen, Haibo and Xing, Wei and Lu, Dongming},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}
```

## Acknowledgement:

We refer to some codes and ideas from [AdaIN](https://github.com/naoto0804/pytorch-AdaIN) and [DIN](https://ojs.aaai.org/index.php/AAAI/article/view/5862). Great thanks to them!
