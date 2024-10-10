from torch import nn
import torch.nn.functional as F
import torch

class DiceLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(DiceLoss, self).__init__()
        # self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # Apply sigmoid or softmax activation to inputs depending on your requirement
        inputs = torch.sigmoid(inputs)

        # Create a one-hot encoded version of the target tensor
        N, C, H, W = inputs.size()
        mask = targets != self.ignore_index
        # make 255 to 0
        targets = targets * mask
        targets_one_hot = torch.zeros(N, C, H, W).to(inputs.device)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)

       # Apply ignore_index
        targets_one_hot = targets_one_hot * mask.unsqueeze(1)

        # Compute per channel Dice Coefficient
        intersection = torch.sum(inputs * targets_one_hot, dim=(2, 3))
        union = torch.sum(inputs, dim=(2, 3)) + torch.sum(targets_one_hot, dim=(2, 3)) + 1e-6

        # Dice score
        dice_score = (2. * intersection + 1e-6) / union
        dice_loss = 1 - dice_score

        # Average over batch and channels
        return dice_loss.mean()
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, ignore_index=255, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        valid_mask = (targets != self.ignore_index)
        BCE_loss = BCE_loss[valid_mask]
        targets = targets[valid_mask]
        inputs = inputs[valid_mask]

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
        
class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5, ce_ratio=0.5, ignore_index=255):
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.ce_ratio = ce_ratio
        self.ignore_index = ignore_index
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, inputs, targets):
        dice_loss = self.dice_loss(inputs, targets)
        ce_loss = self.ce_loss(inputs, targets)
        return self.alpha * dice_loss + self.ce_ratio * ce_loss