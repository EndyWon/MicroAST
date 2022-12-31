import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from torchvision.utils import save_image

import net_microAST as net
from sampler import InfiniteSamplerWrapper

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images',
                    default='./coco2014/train2014') 
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images',
                    default='./wikiart/train') 

parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--sample_path', type=str, default='samples', 
                    help='Derectory to save the intermediate samples')

# training options
parser.add_argument('--save_dir', default='./exp',
                    help='Directory to save the models')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--style_weight', type=float, default=3.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--SSC_weight', type=float, default=3.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--resume', action='store_true', help='train the model from the checkpoint')
parser.add_argument('--checkpoints', default='./checkpoints',
                    help='Directory to save the checkpoint')
args = parser.parse_args()


device = torch.device('cuda:%d' % args.gpu_id)
save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
log_dir = Path(args.log_dir)
log_dir.mkdir(exist_ok=True, parents=True)
checkpoints_dir = Path(args.checkpoints)
checkpoints_dir.mkdir(exist_ok=True, parents=True)
writer = SummaryWriter(log_dir=str(log_dir))

vgg = net.vgg
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

content_encoder = net.Encoder()
style_encoder = net.Encoder()
modulator = net.Modulator()
decoder = net.Decoder()

network = net.Net(vgg, content_encoder, style_encoder, modulator, decoder)
network.train()
network.to(device)

content_tf = train_transform()
style_tf = train_transform()

content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))


optimizer = torch.optim.Adam([
    {'params':network.content_encoder.parameters()}, 
    {'params':network.style_encoder.parameters()}, 
    {'params':network.modulator.parameters()},
    {'params':network.decoder.parameters()}
    ], lr=args.lr)

start_iter = -1

# continue training from the checkpoint
if args.resume:
    checkpoints = torch.load(args.checkpoints + '/checkpoints.pth.tar')
    network.load_state_dict(checkpoints['net'])
    optimizer.load_state_dict(checkpoints['optimizer'])
    start_iter = checkpoints['epoch']

# training
for i in tqdm(range(start_iter+1, args.max_iter)):
    adjust_learning_rate(optimizer, iteration_count=i)
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)
    stylized_results, loss_c, loss_s, loss_contrastive = network(content_images, style_images)
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss_contrastive = args.SSC_weight * loss_contrastive
    loss = loss_c + loss_s + loss_contrastive

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    writer.add_scalar('loss_content', loss_c.item(), i + 1)
    writer.add_scalar('loss_style', loss_s.item(), i + 1)
    writer.add_scalar('loss_contrastive', loss_contrastive.item(), i + 1)
    
    ############################################################################
    # save intermediate samples
    output_dir = Path(args.sample_path)
    output_dir.mkdir(exist_ok=True, parents=True)
    if (i + 1) % 500 == 0: 
        visualized_imgs = torch.cat([content_images, style_images, stylized_results])
        
        output_name = output_dir / 'output{:d}.jpg'.format(i + 1)
        save_image(visualized_imgs, str(output_name), nrow=args.batch_size)
        print('[%d/%d] loss_content:%.4f, loss_style:%.4f, loss_contrastive:%.4f' \
               % (i+1, args.max_iter, loss_c.item(), loss_s.item(), loss_contrastive.item()))    
    ############################################################################

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = network.content_encoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir /
                   'content_encoder_iter_{:d}.pth.tar'.format(i + 1))
        
        state_dict = network.style_encoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir /
                   'style_encoder_iter_{:d}.pth.tar'.format(i + 1))
        
        state_dict = network.modulator.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir /
                   'modulator_iter_{:d}.pth.tar'.format(i + 1))

        state_dict = network.decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir /
                   'decoder_iter_{:d}.pth.tar'.format(i + 1))

        checkpoints = {
            "net": network.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": i
        }
        torch.save(checkpoints, checkpoints_dir / 'checkpoints.pth.tar')  
        
writer.close()

