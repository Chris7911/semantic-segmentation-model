import os
import numpy as np
import cv2
import torch
import imgaug
import json
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


from dataset import ApolloDataset
from model.model_Unet import Resnet34Unet
from model.model_deeplabv3 import DeeplabV3
from utils_dataset import color2label, trainId2label
from utils_metrics import eval_metrics



def score(pred, gt, cls_num):

    total_inter, total_union = 0, 0
    total_correct, total_label = 0, 0

    seg_metrics = eval_metrics(pred, gt, cls_num)
    total_correct += seg_metrics[0]
    total_label += seg_metrics[1]
    total_inter += seg_metrics[2]
    total_union += seg_metrics[3]

    pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
    IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
    mIoU = IoU.mean()
    _get_seg_metrics = {
        "Pixel_Accuracy": np.round(pixAcc, 3),
        "Mean_IoU": np.round(mIoU, 3),
        "Class_IoU": dict(zip(range(seg_cls), np.round(IoU, 3)))
    }

    pixAcc, mIoU, _ = _get_seg_metrics.values()
    

    return pixAcc, mIoU

def draw_image(mask):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)

    # mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    for k, v in color2label.items():
        idx = mask == v.trainId
        r[idx] = k[0]
        g[idx] = k[1]
        b[idx] = k[2]
    color_mask = np.stack([r, g, b], axis=2)
    return color_mask
    
def infer_video(model, video_tfs, device, video_path, save_path):
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (512, 416))
    cam = cv2.VideoCapture(video_path)
    while True:
        ret, img = cam.read()
        vis = np.copy(img)
        img = cv2.cvtColor(vis,cv2.COLOR_BGR2RGB)

        img = img / 255.
        img = video_tfs(img).unsqueeze(0).to(device, dtype=torch.float)
        pred_raw = model(img)
        pred_raw_array = pred_raw.data.cpu().numpy()
        pred_label_imgs = np.argmax(pred_raw_array, axis=1) # (shape: (batch_size, img_h, img_w))
        pred_label_imgs = pred_label_imgs.astype(np.uint8)[0]
        
        pred_img = draw_image(pred_label_imgs)

        resize_vis = cv2.resize(vis, (320, 256))
        # out.write(pred_img)

        cv2.imshow("eval", pred_img)
        
        cv2.imshow('getCamera', resize_vis)

        if 0xFF & cv2.waitKey(5) == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
    assert(0)


def GetModel(arch_type="DeepLab", ckpt_path = None, num_cls=None):
    if arch_type == "Unet":
        ckpt = torch.load(ckpt_path)
        model = Resnet34Unet(3, num_cls)
        model.load_state_dict(ckpt['model_state_dict'])
        return model

    elif arch_type == "DeepLab":
        model = DeeplabV3("eval", "./", num_cls)
        return model


if __name__ == "__main__":
    with open("./Apollo_train.json", "r") as f:
        trainSet = json.load(f)

    with open("./Apollo_valid.json", "r") as f:
        validSet = json.load(f)

    print(len(trainSet), len(validSet))

    train_seq = iaa.Sequential([iaa.size.Resize({"height": 256, "width": 320}, interpolation='nearest'),
                                iaa.Fliplr(0.5)
                                ])
                                
    valid_seq = iaa.Sequential([iaa.size.Resize({"height": 256, "width": 320}),
                                ])

    train_dataset = ApolloDataset(trainSet, seq=train_seq)
    val_dataset = ApolloDataset(validSet, seq=train_seq)
    
    val_loader = DataLoader(dataset=val_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=0)

    
    
    video_tfs = transforms.Compose([valid_seq.augment_image,
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                    std=(0.229, 0.224, 0.225))
                        ])

    device = "cuda"
    seg_cls = val_dataset.get_num_cls()

    model = GetModel(arch_type = "Unet",
                    ckpt_path = './training_logs/model_R34Unet_A/checkpoints/model_R34Unet_A_epoch_10.pth',
                    num_cls = seg_cls)
    
    model.to(device)
    model.eval()

    video_path = "./exp/Test3.mp4"
    save_path = './exp/{}_CE_DiceLoss.avi'.format(os.path.splitext(video_path)[0].split('/')[-1])
    infer_video(model, video_tfs, device, video_path, save_path)


    
    from PIL import Image
    s_pixAcc, s_mIoU, s_iou = 0, 0, 0
    for img, label in val_loader:
        img = img.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.long)
        
        pred_raw = model(img)

        pred_raw_array = pred_raw.data.cpu().numpy()
        pred_label_imgs = np.argmax(pred_raw_array, axis=1) # (shape: (batch_size, img_h, img_w))
        pred_label_imgs = pred_label_imgs.astype(np.uint8)[0]

        pred_label_img_color = draw_image(pred_label_imgs)

        # original image
        img = img[0].data.cpu().numpy()
        img = np.transpose(img, (1, 2, 0)) # (shape: (img_h, img_w, 3))
        img = img*np.array([0.229, 0.224, 0.225])
        img = img + np.array([0.485, 0.456, 0.406])
        img = img*255.0
        img = img.astype(np.uint8)

        label_array = label[0].cpu().numpy()
        label_imgs = label_array.astype(np.uint8)
        label_img_color = draw_image(label_imgs)


        cv2.imshow("original_img", img)
        cv2.imshow("label_img_color", label_img_color)
        cv2.imshow("pred_label_img_color", pred_label_img_color)
        cv2.imwrite("./A_img.png", img)
        cv2.imwrite("./A_label.png", label_img_color)
        cv2.imwrite("./A_img_pred.png", pred_label_img_color)
        
        
        def Iou():
            gt = [(label.cpu() == v) for v in range(seg_cls)]
            gt =  torch.from_numpy(np.stack(gt, axis=1).astype('float')).to(device).float()
            activ  = nn.Softmax(dim=1)
            y_pr   = activ(pred_raw)
            y_pr   = (y_pr > 0.5).type(y_pr.dtype)
            intersection = torch.sum(gt * y_pr)
            union = torch.sum(gt) + torch.sum(y_pr) - intersection + 1e-7
            return (intersection + 1e-7) / union
        
        iou = Iou()

        pixAcc, mIoU = score(pred_raw, label, seg_cls)
        print('Acc {:.2f} mIoU {:.2f} IoU {:.2f}'.format(pixAcc, mIoU, iou.item()))
        s_pixAcc += pixAcc
        s_mIoU   += mIoU
        s_iou    += iou.item()
        

        
        cv2.waitKey(0)
    # print('Sum::::  Acc {:.2f} mIoU {:.2f} IoU {:.2f}'.format(s_pixAcc, s_mIoU, s_iou))
        
