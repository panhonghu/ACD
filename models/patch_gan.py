import torch
import torch.nn as nn
from torch.nn import functional as F
from util import init_model, get_norm_layer


class PatchGAN(nn.Module):
    """PatchGAN discriminator."""
    def __init__(self, args, return_binary=False):
        """Constructs a basic PatchGAN convolutional discriminator.
        Each position in the output is a score of discriminator confidence that
        a 70x70 patch of the input is real.
        Args:
            args: Arguments passed in via the command line.
        """
        super(PatchGAN, self).__init__()
        self.return_binary = return_binary
        norm_layer = get_norm_layer(args.norm_type)
        layers = []
        # Double channels for conditional GAN (concatenated src and tgt images)
        num_channels = args.num_channels
        layers += [nn.Conv2d(num_channels, args.num_channels_d, args.kernel_size_d, stride=2, padding=1),
                   nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(args.num_channels_d, 2 * args.num_channels_d, args.kernel_size_d, stride=2, padding=1),
                   norm_layer(2 * args.num_channels_d),
                   nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(2 * args.num_channels_d, 4 * args.num_channels_d, args.kernel_size_d, stride=2, padding=1),
                   norm_layer(4 * args.num_channels_d),
                   nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(4 * args.num_channels_d, 8 * args.num_channels_d, args.kernel_size_d, stride=1, padding=1),
                   norm_layer(8 * args.num_channels_d),
                   nn.LeakyReLU(0.2, True)]
        if self.return_binary:
            layers += [nn.Conv2d(8 * args.num_channels_d, 1, args.kernel_size_d, stride=1, padding=1)]
        self.model = nn.Sequential(*layers)
        init_model(self.model, init_method=args.initializer)

    def forward(self, input_):
        out_feat = self.model(input_)
        out_feat = F.avg_pool2d(out_feat, out_feat.size()[2:])
        out_feat = out_feat.view(out_feat.size()[0], -1)
        if self.return_binary:
            return out_feat
        else:
            return F.normalize(out_feat, p=2.0, dim=-1)


class SaimeseGAN(nn.Module):
    """PatchGAN discriminator."""
    def __init__(self, args):
        """Constructs a basic PatchGAN convolutional discriminator.
        Each position in the output is a score of discriminator confidence that
        a 70x70 patch of the input is real.
        Args:
            args: Arguments passed in via the command line.
        """
        super(SaimeseGAN, self).__init__()
        norm_layer = get_norm_layer(args.norm_type)
        layers1 = []
        layers2 = []
        # Double channels for conditional GAN (concatenated src and tgt images)
        num_channels = args.num_channels
        layers1 += [nn.Conv2d(num_channels, args.num_channels_d, args.kernel_size_d, stride=2, padding=1),
                   nn.LeakyReLU(0.2, True)]
        layers1 += [nn.Conv2d(args.num_channels_d, 2 * args.num_channels_d, args.kernel_size_d, stride=2, padding=1),
                   norm_layer(2 * args.num_channels_d),
                   nn.LeakyReLU(0.2, True)]
        layers1 += [nn.Conv2d(2 * args.num_channels_d, 4 * args.num_channels_d, args.kernel_size_d, stride=2, padding=1),
                   norm_layer(4 * args.num_channels_d),
                   nn.LeakyReLU(0.2, True)]
        layers1 += [nn.Conv2d(4 * args.num_channels_d, 8 * args.num_channels_d, args.kernel_size_d, stride=1, padding=1),
                   norm_layer(8 * args.num_channels_d),
                   nn.LeakyReLU(0.2, True)]
        layers2 += [nn.Conv2d(num_channels, args.num_channels_d, args.kernel_size_d, stride=2, padding=1),
                   nn.LeakyReLU(0.2, True)]
        layers2 += [nn.Conv2d(args.num_channels_d, 2 * args.num_channels_d, args.kernel_size_d, stride=2, padding=1),
                   norm_layer(2 * args.num_channels_d),
                   nn.LeakyReLU(0.2, True)]
        layers2 += [nn.Conv2d(2 * args.num_channels_d, 4 * args.num_channels_d, args.kernel_size_d, stride=2, padding=1),
                   norm_layer(4 * args.num_channels_d),
                   nn.LeakyReLU(0.2, True)]
        layers2 += [nn.Conv2d(4 * args.num_channels_d, 8 * args.num_channels_d, args.kernel_size_d, stride=1, padding=1),
                   norm_layer(8 * args.num_channels_d),
                   nn.LeakyReLU(0.2, True)]
        self.model1 = nn.Sequential(*layers1)
        self.model2 = nn.Sequential(*layers2)
        self.binary_layer = nn.Sequential(*[nn.Conv2d(16 * args.num_channels_d, 1, args.kernel_size_d, stride=1, padding=1)])
        init_model(self.model1, init_method=args.initializer)
        init_model(self.model2, init_method=args.initializer)
        init_model(self.binary_layer, init_method=args.initializer)

    def forward(self, input1, input2):
        out_feat1 = self.model1(input1)
        out_feat2 = self.model2(input2)
        out_feat = torch.cat((out_feat1, out_feat2), dim=1)
        out_binary = self.binary_layer(out_feat)
        out_binary = F.avg_pool2d(out_binary, out_binary.size()[2:])
        out_binary = out_binary.view(out_binary.size()[0], -1)
        return out_binary


def contour_extraction(imgs):
    imgs = torch.mean(imgs, dim=1)
    iimgs = []
    for img in imgs:
        img_f = torch.fft.fft2(img)
        img_fshift = torch.fft.fftshift(img_f)
        rows, cols = img.shape
        crow, ccol = int(rows/2), int(cols/2)
        img_fshift[crow-5:crow+5, ccol-5:ccol+5] = 0
        img_ishift = torch.fft.ifftshift(img_fshift)
        iimg = torch.fft.ifft2(img_ishift)
        iimg = torch.abs(iimg)
        iimgs.append(iimg.unsqueeze(0))
    return torch.cat(iimgs, 0)

