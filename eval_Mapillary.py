import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
import cv2
import torch
import imgaug
import json
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


from model.model_deeplabv3 import DeeplabV3, Resnet34Unet
from dataset import MapillaryTrainValidSet, MapillaryDataset, Mapillary_labels, UnNormalize

from utils_metrics import eval_metrics, Evaluator



def score(pred_raw, label):
    """PixAcc"""
    predict = torch.argmax(pred_raw.long(), 1) + 1
    # predict = pred_raw.long() + 1

    target = label.long() + 1
    pixel_labeled = torch.sum(target > 0).item()
    pixel_correct = torch.sum((predict == target) * (target > 0)).item()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"

    """mIou"""
    pred_raw = pred_raw.data.cpu().numpy()

    pred_label_imgs = np.argmax(pred_raw, axis=1)
    # pred_label_imgs = pred_raw



    label = label.data.cpu().numpy()
    ious = []

    for i in range(seg_cls):
        pred_inds = pred_label_imgs == i
        target_inds = label == i
        # Cast to long to prevent overflows
        intersection = (pred_inds[target_inds]).sum()
        union = pred_inds.sum() + target_inds.sum() - intersection

        if union == 0:
            iou = float('nan') # If there is no ground truth, do not include in evaluation
        else:
            iou = float(intersection) / float(max(union, 1))
        ious.append(iou)
    
    miou = np.nanmean(ious)
    return pixel_correct, pixel_labeled, miou


def draw_image(img):
    img_height, img_width = img.shape

    img_color = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    for label in Mapillary_labels:
        obj_isinstance = label.hasInstanceignoreInEval
        # if obj_isinstance:
        color = label.color
        trainId = label.trainId
        img_color[img == trainId] = color
        
    return img_color
    
def infer_video(model, video_tfs, device, video_path, save_path):
    count = 0
    img_h = 416
    img_w = 512
    cam = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"MJPG"), 20, (2*img_w, 2*img_h))
    pred_images_list = []
    origin_images_list = []
    while True:
        count += 1
        ret, img = cam.read()
        if ret == True:
            vis = np.copy(img)
            img = cv2.cvtColor(vis,cv2.COLOR_BGR2RGB)

            img = img/255.
            img = video_tfs(img).unsqueeze(0).to(device, dtype=torch.float)
            pred_raw = model(img)
            pred_y = pred_raw.max(1)[1].cpu().data[0].numpy()
            

            pred_y = pred_y.astype(np.int8)
            seg_pred = SegmentationMapsOnImage(pred_y, shape=(416, 512, 3))

            pred_img = draw_image(pred_y)

            # writer.write(pred_img.astype(np.uint8))
            resize_vis = cv2.resize(vis, (512, 416))
            overlayed_img = 0.35*resize_vis + 0.65*pred_img
            overlayed_img = overlayed_img.astype(np.uint8)


            pred_images_list.append("./test/{}.png".format(count))
            origin_images_list.append("./test/{}_original.png".format(count))
            # cv2.imshow("eval", pred_img)
            # cv2.imwrite("./test/{}.png".format(count), pred_img)
            # cv2.imwrite("./test/{}_original.png".format(count), resize_vis)
            # cv2.imwrite("./test/{}_overlayed_img.png".format(count), overlayed_img)
            assert img_h == overlayed_img.shape[0] and img_w == overlayed_img.shape[1], "wrong H, W"

            combined_img = np.zeros((2*img_h, 2*img_w, 3), dtype=np.uint8)
            combined_img[0:img_h, 0:img_w] = resize_vis
            combined_img[0:img_h, img_w:(2*img_w)] = pred_img
            combined_img[img_h:(2*img_h), (int(img_w/2)):(img_w + int(img_w/2))] = overlayed_img
            out.write(combined_img)
            # cv2.imshow("combined_img", combined_img)

            # if 0xFF & cv2.waitKey(5) == 27:
            #     break
        else:
            break

    cam.release()


def GetModel(arch_type="DeepLab", ckpt_path = None, num_cls=None, mode=1):
    ckpt = torch.load(ckpt_path)
    print(ckpt.keys())
    print("weight's epoch: {}".format(ckpt["epoch"]))
    if arch_type == "Unet":
        model = Resnet34Unet(3, num_cls)
        model.load_state_dict(ckpt['mode l_state_dict'])
        return model

    elif arch_type == "DeepLab":
        model = DeeplabV3("eval", "./", num_cls, mode=mode)
        model.load_state_dict(ckpt['model_state_dict'])

        return model


def Compute_mIou(ckpt_folder, val_loader, valid_seq, seg_cls):
    listOfckpt = [ os.path.join(ckpt_folder, ckpt) for ckpt in os.listdir(ckpt_folder)]
    
    seg_cls = val_dataset.get_num_cls()
    model = DeeplabV3("eval", "./", seg_cls, mode=1)

    mertics_miou = np.zeros((len(listOfckpt) + 1))
    mertics_pixAcc = np.zeros((len(listOfckpt) + 1))
    
    for ckpt_path in listOfckpt:
        epoch_version = int(os.path.split(ckpt_path)[-1].split("_")[-1].split(".")[0])
        
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to("cuda")
        model.eval()

        total_miou, total_count = 0, 0
        total_correct, total_label = 0, 0
        for img, label in val_loader:
            torch.cuda.empty_cache()
            img = img.to("cuda", dtype=torch.float)
            label = label.to("cuda", dtype=torch.long)
            
            pred_raw = model(img)
            print(pred_raw.shape)
            print(label.shape)

            pixel_correct, pixel_labeled, miou = score(pred_raw, label)


            total_correct += pixel_correct
            total_label += pixel_labeled
            total_miou += miou
            total_count += 1

        pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
        mIou = 1.0 * total_miou / total_count 
        mertics_miou[epoch_version] = mIou
        mertics_pixAcc[epoch_version] = pixAcc

        plt.figure(1)
        plt.plot(mertics_miou[1:], "k^")
        plt.plot(mertics_miou[1:], "k")
        plt.ylabel("mIou")
        plt.xlabel("epoch")
        plt.title("mIou per epoch")
        plt.savefig("./epoch_mIou_val.png")

        plt.close(1)
        plt.figure(1)
        plt.plot(mertics_pixAcc[1:], "k^")
        plt.plot(mertics_pixAcc[1:], "k")
        plt.ylabel("pixAcc")
        plt.xlabel("epoch")
        plt.title("pixAcc per epoch")
        plt.savefig("./epoch_pixAcc_val.png")
        plt.close(1)
        print('Epoch {}  ::::  Acc {:.2f} mIoU {:.2f} '.format(epoch_version, pixAcc, mIou))



if __name__ == "__main__":



    with open("./Mapillary_validation.json", "r") as f:
        validSet = json.load(f)

    print(len(validSet))
    new_img_h = 416
    new_img_w = 512

                                
    valid_seq = iaa.Sequential([iaa.size.Resize({"height": new_img_h, "width": new_img_w}),
                                ])

    
    val_dataset = MapillaryDataset(validSet, seq=valid_seq)

    val_loader = DataLoader(dataset=val_dataset,
                                batch_size=8,
                                shuffle=False,
                                num_workers=0)
    
    
    video_tfs = transforms.Compose([valid_seq.augment_image,
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                    std=(0.229, 0.224, 0.225))
                        ])
    device = "cuda"
    seg_cls = val_dataset.get_num_cls()
    # ckpt_folder = './training_logs/model_DeepLab_m1_M_CrossEntropyLoss/checkpoints'
    # print("number of class", seg_cls)
    # Compute_mIou(ckpt_folder, val_loader, valid_seq, seg_cls)
    # assert(0)

    folder = "DeepLab_m4_M_CED_SGD"
    # folder = "model_DeepLab_m1_M_CE_DiceLoss"
    # folder = "DeepLab_m1_CED_AUG2"

    # ckpt_path = "./training_logs/{0}/checkpoints/{0}_epoch_30.pth".format(folder)
    ckpt_path = './training_logs/{}/checkpoints/last.pth'.format(folder)
    ckpt_path = './training_logs/{}/checkpoints/best_train.pth'.format(folder)
    ckpt_path = './training_logs/{}/checkpoints/best_valid.pth'.format(folder)
    model = GetModel(arch_type = "DeepLab",
                    ckpt_path = ckpt_path,
                    num_cls = seg_cls,
                    mode = 4)
    
    model.to(device)
    model.eval()

    evaluator = Evaluator(seg_cls)
    # ''' Test Single Image'''
    # img = cv2.imread('./testimg.jpg', -1)
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img_rgb = img_rgb / 255.0
    # img_rgb = img_rgb.astype(np.float32)
    # img_transform = transforms.Compose([transforms.ToTensor(),
    #                                     transforms.Normalize(mean=(0.485, 0.456, 0.406),
    #                                                         std=(0.229, 0.224, 0.225))
    #                                             ])
    # transform_img = valid_seq(image=img_rgb)
    # transform_img = torch.unsqueeze(img_transform(transform_img), 0)
    # print(transform_img.shape)
    # transform_img = transform_img.to(device, dtype=torch.float)
    # pred_raw = model(transform_img)
    # pred_raw = pred_raw.data.cpu().numpy()
    # pred_label_imgs = np.argmax(pred_raw, axis=1) # (shape: (batch_size, img_h, img_w))
    # for idx, label in enumerate(Mapillary_labels):
    #     if label.id in np.unique(pred_label_imgs):
    #         print(label.trainId, " => ", label.name)
    # pred_label_imgs = pred_label_imgs.astype(np.uint8)[0]
    # pred_label_img_color = draw_image(pred_label_imgs)

    # resize_vis = cv2.resize(img, (512, 416))
    # overlayed_img = 0.35*resize_vis + 0.65*pred_label_img_color
    # overlayed_img = overlayed_img.astype(np.uint8)

    # combined_img                           = np.zeros((2*new_img_h, 2*new_img_w, 3), dtype=np.uint8)
    # combined_img[0:new_img_h, 0:new_img_w]         = resize_vis
    # combined_img[0:new_img_h, new_img_w:(2*new_img_w)] = pred_label_img_color
    # combined_img[new_img_h:(2*new_img_h), (int(new_img_w/2)):(new_img_w + int(new_img_w/2))] = overlayed_img

    # cv2.imshow("combined_img", combined_img)
    # cv2.waitKey(0)
    # assert(0)

    video_paths = [
                # "./exp/test1.mp4",
                "./exp/Test3.mp4",
                "./exp/20200806_experiment1.avi",
                "./exp/20200806_experiment2.avi",
                "./exp/FisheyeCamera_1.avi"]

    # for video_path in video_paths:
    #     print("Predict: ", video_path)
    #     ckpt_name = os.path.splitext(ckpt_path.split("/")[-1])[0]
    #     save_path = './exp/{}_{}_{}.avi'.format(os.path.splitext(video_path)[0].split('/')[-1], folder, ckpt_name)
    #     # if not os.path.exists(save_path):
    #     infer_video(model, video_tfs, device, video_path, save_path)
    # assert(0)




    for img, label in val_loader:
        img = img.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.long)
        
        pred_raw = model(img)
        pred_raw = pred_raw.data.cpu().numpy()
        pred_label_imgs = np.argmax(pred_raw, axis=1) # (shape: (batch_size, img_h, img_w))

        label = label.data.cpu().numpy()
        evaluator.add_batch(label, pred_label_imgs)

    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

    print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
    #     pred_label_imgs = pred_label_imgs.astype(np.uint8)[0]

    #     pred_label_img_color = draw_image(pred_label_imgs)
    #     cv2.imshow("pred_label_img_color", pred_label_img_color)


    #    # original image
    #     img = img[0].data.cpu().numpy()
    #     img = np.transpose(img, (1, 2, 0)) # (shape: (img_h, img_w, 3))
    #     img = img*np.array([0.229, 0.224, 0.225])
    #     img = img + np.array([0.485, 0.456, 0.406])
    #     img = img*255.0
    #     img = img.astype(np.uint8)
    #     cv2.imshow("raw_img", img)

    #     # original label
    #     label = label.cpu().squeeze().numpy()
    #     label_image = draw_image(label)
        
    #     cv2.imshow("label", label_image)

    #     cv2.imwrite("./M_img.png", img)
    #     cv2.imwrite("./M_pred.png", pred_label_img_color)
    #     cv2.imwrite("./M_label.png", label_image)
    #     cv2.waitKey(0)
        