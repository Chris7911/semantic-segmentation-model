import os
# os.environ["CUDA_VISIBLE_DEVICES"]="5,6,7"
import numpy as np
import cv2
import json
import torch
import imgaug
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from imgaug import augmenters as iaa

from lr_scheduler import Poly
# from model.model_Unet import Resnet34Unet
from model.model_deeplabv3 import DeeplabV3, Resnet34Unet
from dataset import ApolloDataset, Get_ApolloTrainValidSet
from utils import add_weight_decay
from losses import DiceLoss, CE_DiceLoss, CrossEntropyLoss2d, LovaszSoftmax

def worker_init_fn(worker_id):
    imgaug.seed(np.random.get_state()[1][0] + worker_id)

if __name__ == "__main__":


    batch_size = 8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dir = "/global_share/dataset/road02_seg"
    if not os.path.exists("./Apollo_dataset.json"):
        dataset = Get_ApolloTrainValidSet(data_dir)
    
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


    num_train_batches = int(len(train_dataset)/batch_size)
    num_val_batches = int(len(val_dataset)/batch_size)
    print ("num_train_batches:", num_train_batches)
    print ("num_val_batches:", num_val_batches)

    train_loader = DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=4,
                                worker_init_fn = worker_init_fn)
    
    val_loader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=0)
    
    
    seg_cls = train_dataset.get_num_cls()
    print("num_classes: ", seg_cls)


    model_id = "R34Unet_A"
    '''
    mode == 0: ResNet18_OS16()
    mode == 1: ResNet34_OS16()
    mode == 2: ResNet18_OS8()
    mode == 3: ResNet34_OS8()
    mode == 4: ResNet50_OS16()
    mode == 5: ResNet101_OS16()
    mode == 6: ResNet152_OS16()
    '''
    # model = DeeplabV3(model_id, "./", seg_cls)
    model = Resnet34Unet(model_id, "./", seg_cls)
    model.to(device)


    # trainable_params = filter(lambda p:p.requires_grad, model.parameters())
    trainable_params = add_weight_decay(model, l2_value=0.0001)

    params = {
        "lr": 0.001,
        # "weight_decay": 0.001
        # "momentum": 0.9,
        # "nesterov": True,
    }
    optimizer = torch.optim.Adam(params=trainable_params, **params)
    # optimizer = torch.optim.SGD(params=trainable_params, **params)

    with open("./{}_Apollo_train_class_weights.pkl".format(seg_cls), "rb") as file:
        class_weights = np.array(pickle.load(file))

    class_weights = torch.from_numpy(class_weights) 
    class_weights = class_weights.type(torch.FloatTensor).to(device)


    
    # criterion = CrossEntropyLoss2d(weight=class_weights, reduction="mean")
    # criterion = DiceLoss()
    criterion = CE_DiceLoss(reduction="mean", weight=class_weights)
    # criterion = LovaszSoftmax()

    ''' load weights '''
    isLoad = True
    init_epoch = 0
    if isLoad:
        print("Load weights !!")
        ckpt_path = os.path.join(model.checkpoints_dir, "model_R34Unet_A_epoch_12.pth")
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model_state_dict'])
        init_epoch = ckpt["epoch"]

    
    num_epochs = init_epoch + 20

    iters_per_epoch = len(train_loader)
    lr_scheduler = Poly(optimizer, num_epochs, iters_per_epoch)

    epoch_losses_train = []
    epoch_losses_val = []
    if isLoad and os.path.exists("%s/epoch_losses_val.pkl" % model.model_dir) and\
          os.path.exists("%s/epoch_losses_train.pkl" % model.model_dir):
        with open("%s/epoch_losses_train.pkl" % model.model_dir, "rb") as file:
            epoch_losses_train = list(np.array(pickle.load(file)))

        with open("%s/epoch_losses_val.pkl" % model.model_dir, "rb") as file:
            epoch_losses_val = list(np.array(pickle.load(file)))
    

    for epoch in range(init_epoch, num_epochs):
        print ("epoch: %d/%d" % (epoch+1, num_epochs))

        ############################################################################
        # train:
        ############################################################################
        model.train() # (set in training mode, this affects BatchNorm and dropout)
        batch_losses = []
        for step, (imgs, label_imgs) in enumerate(train_loader):
            imgs = imgs.cuda() # (shape: (batch_size, 3, img_h, img_w))
            label_imgs = label_imgs.type(torch.LongTensor).cuda() # (shape: (batch_size, img_h, img_w))
            outputs = model(imgs)

            # compute the loss:
            loss = criterion(outputs, label_imgs)
            loss_value = loss.data.cpu().numpy()
            batch_losses.append(loss_value)

            # optimization step:
            optimizer.zero_grad() # (reset gradients)
            loss.backward() # (compute gradients)
            optimizer.step() # (perform optimization step)


        epoch_loss = np.mean(batch_losses)
        epoch_losses_train.append(epoch_loss)
        with open("%s/epoch_losses_train.pkl" % model.model_dir, "wb") as file:
            pickle.dump(epoch_losses_train, file)
        TRAINLOSS_str = "train loss: {}".format(epoch_loss)
        plt.figure(1)
        plt.plot(epoch_losses_train, "k^")
        plt.plot(epoch_losses_train, "k")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.title("train loss per epoch")
        plt.savefig("%s/epoch_losses_train.png" % model.model_dir)
        plt.close(1)

        ############################################################################
        # val:
        ############################################################################
        model.eval() # (set in evaluation mode, this affects BatchNorm and dropout)
        batch_losses = []

        for step, (imgs, label_imgs) in enumerate(val_loader):
            with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
                imgs = imgs.cuda() # (shape: (batch_size, 3, img_h, img_w))
                label_imgs = label_imgs.type(torch.LongTensor).cuda() # (shape: (batch_size, img_h, img_w))

                outputs = model(imgs) # (shape: (batch_size, num_classes, img_h, img_w))

                # compute the loss:
                loss = criterion(outputs, label_imgs)
                loss_value = loss.data.cpu().numpy()
                batch_losses.append(loss_value)

        epoch_loss = np.mean(batch_losses)
        epoch_losses_val.append(epoch_loss)
        with open("%s/epoch_losses_val.pkl" % model.model_dir, "wb") as file:
            pickle.dump(epoch_losses_val, file)
        VALIDLOSS_str = "val loss: {}".format(epoch_loss)
        plt.figure(1)
        plt.plot(epoch_losses_val, "k^")
        plt.plot(epoch_losses_val, "k")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.title("val loss per epoch")
        plt.savefig("%s/epoch_losses_val.png" % model.model_dir)
        plt.close(1)

        # save the model weights to disk:

        print("\t" + TRAINLOSS_str + " , " + VALIDLOSS_str)
        checkpoint_path = model.checkpoints_dir + "/model_" + model_id +"_epoch_" + str(epoch+1) + ".pth"
        torch.save({ 'epoch': epoch, 'model_state_dict': model.state_dict()}, checkpoint_path)

