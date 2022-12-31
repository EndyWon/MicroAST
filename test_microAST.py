import argparse
from pathlib import Path

import time
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import net_microAST as net

import traceback
import thop


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str,
                    help='File path to the style image')
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--content_encoder', type=str, default='models/content_encoder_iter_160000.pth.tar')
parser.add_argument('--style_encoder', type=str, default='models/style_encoder_iter_160000.pth.tar')
parser.add_argument('--modulator', type=str, default='models/modulator_iter_160000.pth.tar')
parser.add_argument('--decoder', type=str, default='models/decoder_iter_160000.pth.tar')

# Additional options
parser.add_argument('--content_size', type=int, default=0,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=0,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')
parser.add_argument('--gpu_id', type=int, default=0)

# Advanced options
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')

args = parser.parse_args()

device = torch.device('cuda:%d' % args.gpu_id)

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

# Either --content or --contentDir should be given.
assert (args.content or args.content_dir)
if args.content:
    content_paths = [Path(args.content)]
else:
    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*')]

# Either --style or --styleDir should be given.
assert (args.style or args.style_dir)
if args.style:
    style_paths = [Path(args.style)]
else:
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]

content_encoder = net.Encoder()
style_encoder = net.Encoder()
modulator = net.Modulator()
decoder = net.Decoder()

content_encoder.eval()
style_encoder.eval()
modulator.eval()
decoder.eval()

content_encoder.load_state_dict(torch.load(args.content_encoder))
style_encoder.load_state_dict(torch.load(args.style_encoder))
modulator.load_state_dict(torch.load(args.modulator))
decoder.load_state_dict(torch.load(args.decoder))

network = net.TestNet(content_encoder, style_encoder, modulator, decoder)

network.to(device)

content_tf = test_transform(args.content_size, args.crop)
style_tf = test_transform(args.style_size, args.crop)

for content_path in content_paths:
    for style_path in style_paths:
        try:
            content = content_tf(Image.open(str(content_path)))
            style = style_tf(Image.open(str(style_path)))
            
            style = style.to(device).unsqueeze(0)
            content = content.to(device).unsqueeze(0)
            
            torch.cuda.synchronize()
            tic = time.time()
            
            with torch.no_grad():
                output = network(content, style, args.alpha)
                #flops, params = thop.profile(network, inputs=(content, style, args.alpha))
                #print ("GFLOPS: %.4f, Params: %.4f"% (flops/1e9, params/1e6))
            
            torch.cuda.synchronize()
            print ("Elapsed time: %.4f seconds"%(time.time()-tic))
            #print ("Max GPU memory allocated: %.4f GB" % (torch.cuda.max_memory_allocated(device=args.gpu_id) / 1024. / 1024. / 1024.))
            output = output.cpu()

            output_name = output_dir / '{:s}_stylized_{:s}{:s}'.format(
                content_path.stem, style_path.stem, args.save_ext)
            save_image(output, str(output_name))
        except:
            traceback.print_exc()
