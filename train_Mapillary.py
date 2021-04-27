import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
from tensorboardX import SummaryWriter
from lr_scheduler import Poly
from model.model_deeplabv3 import DeeplabV3, Resnet34Unet
from dataset import MapillaryTrainValidSet, MapillaryDataset, Mapillary_labels
from utils import add_weight_decay
from losses import DiceLoss, CE_DiceLoss, CrossEntropyLoss2d, LovaszSoftmax
# from undistort import distort_augmenter
from torchvision.utils import make_grid
from utils_metrics import Evaluator

def denormalize(images, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    for i in range(images.shape[1]):
        images[:, i, :, :] = images[:, i, :, :] * std[i] + mean[i]       
    
    images = images[:, [2, 1, 0], :, :]
    

    images = images * 255
    return images.byte()

def PlotNumberPic(values, v_name, filepath, plotpath, xlabel="epoch", ylabel="loss"):
    with open(filepath, "wb") as file:
        pickle.dump(values, file)
    plt.figure(1)
    plt.plot(values, "k^")
    plt.plot(values, "k")
    plt.title(v_name)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(plotpath)
    plt.close(1)

def worker_init_fn(worker_id):
    imgaug.seed(np.random.get_state()[1][0] + worker_id)


def draw_label(label_imgs):
    img_height, img_width = label_imgs.shape[1], label_imgs.shape[2]

    decode_imgs = []
    for label_img in label_imgs:
        img_color = np.zeros((img_height, img_width, 3), dtype=np.uint8)


        r = np.zeros_like(label_img).astype(np.uint8)
        g = np.zeros_like(label_img).astype(np.uint8)
        b = np.zeros_like(label_img).astype(np.uint8)
        rgb = np.stack([r,g,b], axis=0)
        

        for label in Mapillary_labels:
            color = label.color
            trainId = label.trainId
            img_color[label_img == trainId] = color
        
        decode_imgs.append(torch.from_numpy(np.transpose(img_color, (2, 0, 1))))
        
    return decode_imgs

if __name__ == "__main__":
    batch_size = 8
    new_img_h = 480
    new_img_w = 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 

    with open("./Mapillary_training.json", "r") as f:
        trainSet = json.load(f)

    with open("./Mapillary_validation.json", "r") as f:
        validSet = json.load(f)

    print(len(trainSet), len(validSet))
    

    train_seq = iaa.Sequential([iaa.size.Resize({"height": new_img_h, "width": new_img_w}, interpolation='nearest'),
                                iaa.Fliplr(0.5),
                                iaa.Multiply((0.8, 1.5)),                               
                                # iaa.OneOf([ 
                                #             iaa.MotionBlur(k=4),
                                #             iaa.GaussianBlur((1.0)),
                                #             iaa.Noop(),
                                #         ]),
                                iaa.Sometimes(0.5, iaa.Sharpen(alpha=(0, 0.8))),
                                # iaa.Sometimes(0.3,  iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                                ])
                                
    valid_seq = iaa.Sequential([iaa.size.Resize({"height": new_img_h, "width": new_img_w}),
                                ])

    train_dataset = MapillaryDataset(trainSet, seq=train_seq)
    val_dataset = MapillaryDataset(validSet, seq=valid_seq)
    # for i in range(len(train_dataset)):

    #     img, label_image = train_dataset[i]
    #     # print(img.shape, label_image.shape)
    #     img = img.numpy()
    #     img = np.transpose(img, (1, 2, 0)) # (shape: (img_h, img_w, 3))
    #     img = img*np.array([0.229, 0.224, 0.225])
    #     img = img + np.array([0.485, 0.456, 0.406])
    #     img = img * 255.0
    #     img = img.astype(np.uint8)
    #     cv2.imshow("raw_img", img)

    #     label_image = label_image.numpy()
    #     color_array = np.zeros((label_image.shape[0], label_image.shape[1], 3), dtype=np.uint8)

    #     for idx, label in enumerate(Mapillary_labels):
    #         if  label.id in np.unique(label_image):
    #             print(label.trainId, " => ", label.name)
    #             color_array[label_image == label.id] = label.color
    #     cv2.imshow("ins", color_array[:, :, ::-1])
    #     cv2.waitKey(0)
        
    #     assert(0)


    num_train_batches = int(len(train_dataset)/batch_size)
    num_val_batches = int(len(val_dataset)/batch_size)
    
    print(len(train_dataset), len(val_dataset))
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


    model_id = "DeepLab_m4_M_CED_SGD"
    '''
    mode == 0: ResNet18_OS16()
    mode == 1: ResNet34_OS16()
    mode == 2: ResNet18_OS8()
    mode == 3: ResNet34_OS8()
    mode == 4: ResNet50_OS16()
    mode == 5: ResNet101_OS16()
    mode == 6: ResNet152_OS16()
    '''
    model = DeeplabV3(model_id, "./", seg_cls, mode=4)
    # model = Resnet34Unet(model_id, "./", seg_cls)
    model.to(device)

    class_weights = None
    # if os.path.exists("./{}_Mapillary_train_class_weights.pkl".format(seg_cls)):
    #     with open("./{}_Mapillary_train_class_weights.pkl".format(seg_cls), "rb") as file:
    #         class_weights = np.array(pickle.load(file))

    #     class_weights = torch.from_numpy(class_weights) 
    #     class_weights = class_weights.type(torch.FloatTensor).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model_dir = model.module.model_dir if hasattr(model, 'module') else model.model_dir
    ckpt_path = model.module.checkpoints_dir if hasattr(model, 'module') else model.checkpoints_dir


    ''' load weights '''
    isLoad = False
    init_epoch = 0
    if isLoad:
        print("Load weights !!")
        # C:\Users\User\Desktop\aigo_seg_model\training_logs\model_R34Unet\checkpoints\model_R34Unet_epoch_8.pth
        ckpt = torch.load(ckpt_path + "/last.pth")
        model.load_state_dict(ckpt['model_state_dict'])
        init_epoch = ckpt["epoch"]

    trainable_params = filter(lambda p:p.requires_grad, model.parameters())
    # trainable_params = add_weight_decay(model, l2_value=0.0001)
    learning_rate = 0.0001
    params = {
        "lr": learning_rate,
        "momentum": 0.9,
        "nesterov": True,
    }
    # optimizer = torch.optim.Adam(params=trainable_params, **params)
    optimizer = torch.optim.SGD(params=trainable_params, **params)

    # criterion = CrossEntropyLoss2d(weight=class_weights, reduction="mean")
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    # criterion = DiceLoss()
    criterion = CE_DiceLoss(reduction="mean", weight=class_weights)
    # criterion = LovaszSoftmax()
    
    num_epochs = init_epoch + 500

    iters_per_epoch = len(train_loader)

    lr_scheduler = Poly(optimizer, num_epochs, iters_per_epoch)


    # Plot lr schedule
    # y = []
    # for epoch in range(init_epoch, num_epochs):
    #     lr_scheduler.step()
    #     print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
    #     y.append(optimizer.param_groups[0]['lr'])
    # plt.plot(y, '.-', label='LambdaLR')
    # plt.xlabel('epoch')
    # plt.ylabel('LR')
    # plt.tight_layout()
    # plt.savefig('./LR.png', dpi=300)

    

    epoch_losses_train = []
    epoch_losses_val = []
    mertics_val_miou = []
    mertics_val_pixAcc = []
    mertics_val_FWIoU = []


    if isLoad and os.path.exists("%s/epoch_losses_val.pkl" % model_dir) and\
        os.path.exists("%s/epoch_losses_train.pkl" % model_dir):

        with open("%s/epoch_losses_train.pkl" % model_dir, "rb") as file:
            epoch_losses_train = list(np.array(pickle.load(file)))

        with open("%s/epoch_losses_val.pkl" % model_dir, "rb") as file:
            epoch_losses_val = list(np.array(pickle.load(file)))

        with open("%s/epoch_miou_val.pkl" % model_dir, "rb") as file:
            mertics_val_miou = list(np.array(pickle.load(file)))

        with open("%s/epoch_FWIoU_val.pkl" % model_dir, "rb") as file:
            mertics_val_FWIoU = list(np.array(pickle.load(file)))
        
        with open("%s/epoch_pixAcc_val.pkl" % model_dir, "rb") as file:
            mertics_val_pixAcc = list(np.array(pickle.load(file)))

        print("Load Prev train Loss: ", len(epoch_losses_train))
        print("Load Prev val Loss: ", len(epoch_losses_val))
        print("Load Prev miou Loss: ", len(mertics_val_miou))
        print("Load Prev pixAcc Loss: ", len(mertics_val_pixAcc))

    
    min_train_loss = np.inf
    min_valid_loss = np.inf
    max_fwiou = -np.inf
    max_pixacc = -np.inf


    tensorboard_path = os.path.join(model_dir, "runs")
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)


    evaluator = Evaluator(seg_cls)

    writer = SummaryWriter(tensorboard_path)

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

        lr_scheduler.step()


        epoch_loss = np.mean(batch_losses)
        epoch_losses_train.append(epoch_loss)
        TRAINLOSS_str = "train loss: {:.4f}".format(epoch_loss)
        writer.add_scalar('train/loss_epoch', epoch_loss, epoch)

        PlotNumberPic(epoch_losses_train,
                    v_name = "train loss per epoch",
                    filepath = "%s/epoch_losses_train.pkl" % model_dir,
                    plotpath = "%s/epoch_losses_train.png" % model_dir,
                    xlabel="epoch",
                    ylabel="loss")


        ############################################################################
        # val:
        ############################################################################
        model.eval() # (set in evaluation mode, this affects BatchNorm and dropout)
        batch_losses = []
        mertics_miou = []
        mertics_pixAcc = []
        
        for step, (imgs, label_imgs) in enumerate(val_loader):
            with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
                imgs = imgs.cuda() # (shape: (batch_size, 3, img_h, img_w))
                label_imgs = label_imgs.type(torch.LongTensor).cuda() # (shape: (batch_size, img_h, img_w))

                outputs = model(imgs) # (shape: (batch_size, num_classes, img_h, img_w))

                # compute the loss:
                loss = criterion(outputs, label_imgs)
                loss_value = loss.data.cpu().numpy()
                batch_losses.append(loss_value)
                
                outputs = outputs.data.cpu().numpy()
                outputs = np.argmax(outputs, axis=1)
                label_imgs = label_imgs.data.cpu().numpy()
                evaluator.add_batch(label_imgs, outputs)

                if step == 0:
                    grid_image = make_grid(denormalize(imgs[:3].data.cpu()), 3)
                    writer.add_image('Image', grid_image, epoch)
                    
                    
                    grid_image = make_grid(draw_label(label_imgs[:3]), 3)
                    writer.add_image('Label', grid_image, epoch)

                    grid_image = make_grid(draw_label(outputs[:3]), 3)
                    writer.add_image('Prediction', grid_image, epoch)

        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

        
        mertics_val_miou.append(mIoU)
        mertics_val_pixAcc.append(Acc)
        mertics_val_FWIoU.append(FWIoU)
        writer.add_scalar('val/Acc_class', Acc_class, epoch)



        MIOU_str = "miou: {:.4f}".format(mIoU)
        writer.add_scalar('val/mIoU', mIoU, epoch)
        PlotNumberPic(mertics_val_miou,
                    v_name = "miou per epoch",
                    filepath = "%s/epoch_miou_val.pkl" % model_dir,
                    plotpath = "%s/epoch_miou_val.png" % model_dir,
                    xlabel="epoch",
                    ylabel="miou")

        FWIoU_str = "FWIoU: {:.4f}".format(FWIoU)
        writer.add_scalar('val/fwIoU', FWIoU, epoch)
        PlotNumberPic(mertics_val_miou,
                    v_name = "FWIoU per epoch",
                    filepath = "%s/epoch_FWIoU_val.pkl" % model_dir,
                    plotpath = "%s/epoch_FWIoU_val.png" % model_dir,
                    xlabel="epoch",
                    ylabel="FWIoU")

        PIXACC_str = "pixAcc: {:.4f}".format(Acc)
        writer.add_scalar('val/Acc', Acc, epoch)
        PlotNumberPic(mertics_val_pixAcc,
                    v_name = "pixAcc per epoch",
                    filepath = "%s/epoch_pixAcc_val.pkl" % model_dir,
                    plotpath = "%s/epoch_pixAcc_val.png" % model_dir,
                    xlabel="epoch",
                    ylabel="pixAcc")

        epoch_loss = np.mean(batch_losses)
        epoch_losses_val.append(epoch_loss)
        VALIDLOSS_str = "val loss: {:.3f}".format(epoch_loss)
        writer.add_scalar('val/loss_epoch', epoch_loss, epoch)
        PlotNumberPic(epoch_losses_val,
                    v_name = "val per epoch",
                    filepath = "%s/epoch_losses_val.pkl" % model_dir,
                    plotpath = "%s/epoch_losses_val.png" % model_dir,
                    xlabel="epoch",
                    ylabel="loss")



        # save the model weights to disk:

        print("\t" + TRAINLOSS_str + " , " + VALIDLOSS_str + " , "\
                + MIOU_str + " , " + FWIoU_str + ' , ' + \
                PIXACC_str + " , lr:{:.8f}".format(optimizer.param_groups[0]['lr']))
        
        checkpoint_path = ckpt_path + "/model_" + model_id +"_epoch_" + str(epoch+1) + ".pth"
        torch.save({ 'epoch': epoch, 
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'lr': optimizer.param_groups[0]['lr']
                    }, ckpt_path + "/last.pth" )
        
        if min_train_loss > epoch_losses_train[-1]:
            min_train_loss = epoch_losses_train[-1]
            torch.save({ 'epoch': epoch, 
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'lr': optimizer.param_groups[0]['lr'],
                    "value": min_train_loss
                    }, ckpt_path + "/best_train.pth" )
        
        if min_valid_loss > epoch_losses_val[-1]:
            min_train_loss = epoch_losses_val[-1]
            torch.save({ 'epoch': epoch, 
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'lr': optimizer.param_groups[0]['lr'],
                    "value": min_valid_loss
                    }, ckpt_path + "/best_valid.pth" )

        if max_fwiou < FWIoU:
            max_fwiou = FWIoU
            torch.save({ 'epoch': epoch, 
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'lr': optimizer.param_groups[0]['lr'],
                    "value": max_fwiou
                    }, ckpt_path + "/best_fwiou.pth" )
        
        if max_pixacc < Acc:
            max_pixacc = Acc
            torch.save({ 'epoch': epoch, 
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'lr': optimizer.param_groups[0]['lr'],
                    "value": max_pixacc
                    }, ckpt_path + "/best_pixacc.pth" )


