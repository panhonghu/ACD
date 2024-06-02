import random
import torch
import torch.nn as nn
from argparse import ArgumentError


def gan_class_name(s):
    """Convert a string to a case-sensitive GAN class name.
    Used to parse gan name from the command line
    """
    s = s.lower().strip().replace('_', '')
    if s == 'pix2pix':
        class_name = 'Pix2Pix'
    elif s == 'flowpix2pix':
        class_name = 'FlowPix2Pix'
    elif s == 'cyclegan':
        class_name = 'CycleGAN'
    elif s == 'cycleflow':
        class_name = 'CycleFlow'
    elif s == 'flow2flow':
        class_name = 'Flow2Flow'
    else:
        raise ArgumentError('Argument does not match a GAN type: {}'.format(s))
    return class_name


class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


class HiddenFeatureLoss(nn.Module):
    def __init__(self, margin=500):
        super(HiddenFeatureLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        dmat = self.compute_euclidean_dist(embeddings, embeddings)   # bs*bs
        # print(dmat[0])
        bs = embeddings.shape[0]
        labels1 = labels.unsqueeze(1).repeat(1, bs)
        labels2 = labels.unsqueeze(0).repeat(bs, 1)
        mask = (labels1==labels2).float()   # only considering features corresponding to the same label
        mask_pos = mask - torch.Tensor(torch.eye(bs)).to(mask.device)
        loss_pos = torch.sum(dmat*mask_pos)/torch.sum(mask_pos)
        mask_neg = 1.0 - mask
        loss_neg = torch.sum(dmat*mask_neg)/torch.sum(mask_neg)
        # return self.margin+loss_pos-loss_neg
        return loss_pos + 10.0/max(1.0, loss_neg)

    def compute_euclidean_dist(self, x, y):
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()
        return dist


class CrossModalityIdentityLoss(nn.Module):
    def __init__(self):
        super(CrossModalityIdentityLoss, self).__init__()

    def forward(self, embeddings1, embeddings2, labels1, labels2):   #bs*512 bs
        assert embeddings1.shape==embeddings2.shape and labels1.shape==labels2.shape, \
                        "check the input for CrossModalityIdentityLoss"
        dmat = self.compute_unit_l2_distance(embeddings1, embeddings2)   # bs*bs
        bs = embeddings1.shape[0]
        labels1 = labels1.unsqueeze(1).repeat(1, bs)
        labels2 = labels2.unsqueeze(0).repeat(bs, 1)
        mask = (labels1==labels2).float()   # only considering features corresponding to the same label
        # print(dmat)
        # print(mask)
        return torch.sum(dmat*mask)/torch.sum(mask)

    def compute_unit_l2_distance(self, a, b, eps=1e-6):
        """
        computes pairwise Euclidean distance and return a N x N matrix
        """
        dmat = torch.matmul(a, torch.transpose(b, 0, 1))
        dmat = ((1.0 - dmat + eps) * 2.0).pow(0.5)
        return dmat


class GANLoss(nn.Module):
    """Module for computing the GAN loss for the generator.
    When `use_least_squares` is turned on, we use mean squared error loss,
    otherwise we use the standard binary cross-entropy loss.
    Note: We use the convention that the discriminator predicts the probability
    that the target image is real. Therefore real corresponds to label 1.0."""
    def __init__(self, device, use_least_squares=False):
        super(GANLoss, self).__init__()
        self.loss_fn = nn.MSELoss() if use_least_squares else nn.BCELoss()
        self.real_label = None  # Label tensor passed to loss if target is real
        self.fake_label = None  # Label tensor passed to loss if target is fake
        self.device = device

    def _get_label_tensor(self, input_, is_tgt_real):
        # Create label tensor if needed
        if is_tgt_real and (self.real_label is None or self.real_label.numel() != input_.numel()):
            self.real_label = torch.ones_like(input_, device=self.device, requires_grad=False)
        elif not is_tgt_real and (self.fake_label is None or self.fake_label.numel() != input_.numel()):
            self.fake_label = torch.zeros_like(input_, device=self.device, requires_grad=False)
        return self.real_label if is_tgt_real else self.fake_label

    def __call__(self, input_, is_tgt_real):
        label = self._get_label_tensor(input_, is_tgt_real)
        return self.loss_fn(input_, label)

    def forward(self, input_):
        raise NotImplementedError('GANLoss should be called directly.')


class JacobianClampingLoss(nn.Module):
    """Module for adding Jacobian Clamping loss.
    See Also:
        https://arxiv.org/abs/1802.08768v2
    """
    def __init__(self, lambda_min=1., lambda_max=20.):
        super(JacobianClampingLoss, self).__init__()
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

    def forward(self, gz, gz_prime, z, z_prime):
        q = (gz - gz_prime).norm() / (z - z_prime).norm()
        l_max = (q.clamp(self.lambda_max, float('inf')) - self.lambda_max) ** 2
        l_min = (q.clamp(float('-inf'), self.lambda_min) - self.lambda_min) ** 2
        l_jc = l_max + l_min
        return l_jc

