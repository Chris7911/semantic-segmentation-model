import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)






class SoftDiceLoss(nn.Module):
    def __init__(self, num_classes, eps=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, preds, targets): #pylint: disable=unused-argument
        """
        è®¡ç®Soft-Dice Loss

        Arguments:
            preds (torch.FloatTensor):
                é¢æµæ ç­¾çtensor. tensorçshapeä¸º(B, num_classes, H, W)
            targets (torch.LongTensor):
                ground-truthæ ç­¾çtensor, shapeä¸º(B, 1, H, W)
        Returns:
            mean_loss (float32): mean loss by class value
        """
        loss = 0
        print(targets.shape, torch.unique(targets))
        for cls in range(self.num_classes):
            target = (targets == cls).float()

            pred = preds[:, cls]
            print(pred.shape, torch.unique(pred))

            intersection = (pred * target).sum()

            dice = (2 * intersection + self.eps)/(pred.sum() + target.sum() + self.eps)

            loss = loss - dice.log()

            loss = loss/self.num_classes

        return loss.item()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.val

    @property
    def average(self):
        return np.round(self.avg, 5)

def eval_metrics(output, target, num_class):
    # _, predict = torch.max(output.data, 1)
    # predict = output.data
    # predict = predict + 1
    # target = target + 1

    # labeled = (target > 0) * (target <= num_class)
    # correct, num_labeled = batch_pix_accuracy(predict, target, labeled)
    # inter, union = batch_intersection_union(predict, target, num_class, labeled)
    # return [np.round(correct, 5), np.round(num_labeled, 5), np.round(inter, 5), np.round(union, 5)]


    correct, labeled = batch_pix_accuracy(output, target)
    inter, union = batch_intersection_union(output, target, num_class)

    total_correct = correct
    total_label = labeled
    # if self.total_inter.device != inter.device:
    #     self.total_inter = self.total_inter.to(inter.device)
    #     self.total_union = self.total_union.to(union.device)
    total_inter = inter
    total_union = union
    pixAcc = 1.0 * total_correct / (2.220446049250313e-16 + total_label)  # remove np.spacing(1)
    IoU = 1.0 * total_inter / (2.220446049250313e-16 + total_union)
    mIoU = IoU.mean().item()
    return pixAcc, mIoU



def batch_pix_accuracy(output, target):
    """PixAcc"""
    # inputs are numpy array, output 4D, target 3D
    predict = torch.argmax(output.long(), 1) + 1
    target = target.long() + 1

    pixel_labeled = torch.sum(target > 0).item()
    pixel_correct = torch.sum((predict == target) * (target > 0)).item()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):
    """mIoU"""
    # inputs are numpy array, output 4D, target 3D
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = torch.argmax(output, 1) + 1  # [N,H,W] 
    target = target.float() + 1            # [N,H,W] 

    predict = predict.float() * (target > 0).float()
    intersection = predict * (predict == target).float()
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    area_inter = torch.histc(intersection.cpu(), bins=nbins, min=mini, max=maxi)
    area_pred = torch.histc(predict.cpu(), bins=nbins, min=mini, max=maxi)
    area_lab = torch.histc(target.cpu(), bins=nbins, min=mini, max=maxi)
    area_union = area_pred + area_lab - area_inter
    assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
    return area_inter.float(), area_union.float()

# def batch_pix_accuracy(predict, target, labeled):
#     pixel_labeled = labeled.sum()
#     pixel_correct = ((predict == target) * labeled).sum()
#     assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
#     return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()

# def batch_intersection_union(predict, target, num_class, labeled):
#     predict = predict * labeled.long()
#     intersection = predict * (predict == target).long()

#     area_inter = torch.histc(intersection.float(), bins=num_class, max=num_class, min=1)
#     area_pred = torch.histc(predict.float(), bins=num_class, max=num_class, min=1)
#     area_lab = torch.histc(target.float(), bins=num_class, max=num_class, min=1)
#     area_union = area_pred + area_lab - area_inter
#     assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
#     return area_inter.cpu().numpy(), area_union.cpu().numpy()

# if __name__ == "__main__":
#     ground_truth = torch.zeros(1, 224, 224)
#     ground_truth[:, :50, :50] = 1
#     ground_truth[:, 50:100, 50:100] = 2

#     prediction = torch.zeros(1, 3, 224, 224).uniform_().softmax(dim=1)
#     print(prediction.shape)

#     soft_dice_loss = SoftDiceLoss(num_classes=3)

#     loss = soft_dice_loss(prediction, ground_truth)

#     print('Loss: {}'.format(loss))