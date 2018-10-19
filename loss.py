import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import utils
from lovasz_losses import lovasz_hinge


class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.nll_loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        return self.nll_loss((1 - F.sigmoid(inputs)) ** self.gamma * F.logsigmoid(inputs), targets)


class FocalLovasz(nn.Module):
    def __init__(self, per_image=True, ignore=None, focal_weight=0.5):
        super().__init__()
        self.focal_loss = RobustFocalLoss2d()
        self.per_image = per_image
        self.ignore = ignore
        self.focal_weight = focal_weight

    def __call__(self, logits, labels):
        return lovasz_hinge(logits, labels, self.per_image, self.ignore) + self.focal_loss(logits,
                                                                                           labels) * self.focal_weight


class FocalJaccard(nn.Module):
    def __init__(self, jaccard_weight=0.3):
        super().__init__()
        self.focal_loss = RobustFocalLoss2d()
        self.jaccard_weight = jaccard_weight

    def __call__(self, logits, labels):
        loss = (1 - self.jaccard_weight) * self.focal_loss(logits, labels)

        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (labels == 1).float()
            jaccard_output = F.sigmoid(logits)

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
        return loss


class RFocalLovaszJaccard(nn.Module):
    def __init__(self, jaccard_weight=0.3, focal_weight=0.3):
        super().__init__()
        self.focal_loss = RobustFocalLoss2d()
        self.jaccard_weight = jaccard_weight
        self.focal_weight = focal_weight

    def __call__(self, logits, labels):
        loss = (1 - self.jaccard_weight) * (
            lovasz_hinge(logits, labels, per_image=True, ignore=None)) + self.focal_loss(logits,
                                                                                         labels) * self.focal_weight

        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (labels == 1).float()
            jaccard_output = F.sigmoid(logits)

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
        return loss


class RobustFocalLoss2d(nn.Module):
    # assume top 10% is outliers
    def __init__(self, gamma=2, size_average=True):
        super().__init__()
        self.gamma = gamma
        self.size_average = size_average

    def __call__(self, logit, target, class_weight=None, type='sigmoid'):
        target = target.view(-1, 1).long()

        if type == 'sigmoid':
            if class_weight is None:
                class_weight = [1] * 2  # [0.5, 0.5]

            prob = F.sigmoid(logit)
            prob = prob.view(-1, 1)
            prob = torch.cat((1 - prob, prob), 1)
            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.)

        elif type == 'softmax':
            B, C, H, W = logit.size()
            if class_weight is None:
                class_weight = [1] * C  # [1/C]*C

            logit = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob = F.softmax(logit, 1)
            select = torch.FloatTensor(len(prob), C).zero_().cuda()
            select.scatter_(1, target, 1.)

        class_weight = torch.FloatTensor(class_weight).cuda().view(-1, 1)
        class_weight = torch.gather(class_weight, 0, target)

        prob = (prob * select).sum(1).view(-1, 1)
        prob = torch.clamp(prob, 1e-8, 1 - 1e-8)

        focus = torch.pow((1 - prob), self.gamma)
        # focus = torch.where(focus < 2.0, focus, torch.zeros(prob.size()).cuda())
        focus = torch.clamp(focus, 0, 2)

        batch_loss = - class_weight * focus * prob.log()

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss

        return loss


class LossBinary:
    """
    Loss defined as \alpha BCE - (1 - \alpha) SoftJaccard
    """

    def __init__(self, jaccard_weight=0.3):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1).float()
            jaccard_output = F.sigmoid(outputs)

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
        return loss


class LossMulti:
    def __init__(self, jaccard_weight=0, class_weights=None, num_classes=1):
        if class_weights is not None:
            nll_weight = utils.cuda(
                torch.from_numpy(class_weights.astype(np.float32)))
        else:
            nll_weight = None
        self.nll_loss = nn.NLLLoss2d(weight=nll_weight)
        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            for cls in range(self.num_classes):
                jaccard_target = (targets == cls).float()
                jaccard_output = outputs[:, cls].exp()
                intersection = (jaccard_output * jaccard_target).sum()

                union = jaccard_output.sum() + jaccard_target.sum()
                loss -= torch.log((intersection + eps) / (union - intersection + eps)) * self.jaccard_weight
        return loss
