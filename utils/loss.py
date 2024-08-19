import torch.nn.functional as F


def LossFunc(pred, mask):
    mask=mask.float()
    bce = F.binary_cross_entropy(pred, mask, reduce=None)

    inter = ((pred * mask)).sum(dim=(2, 3))
    union = ((pred + mask)).sum(dim=(2, 3))
    aiou = 1 - (inter + 1) / (union - inter + 1)

    mae = F.l1_loss(pred, mask, reduce=None)

    return (bce + aiou + mae).mean()

