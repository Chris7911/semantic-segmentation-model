import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch2trt
import time
from model.model_deeplabv3 import DeeplabV3, Resnet34Unet
# from imgaug import augmenters as iaa
# from imgaug.augmentables.segmaps import SegmentationMapsOnImage

from collections import namedtuple
Label = namedtuple('Label', [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!
    'hasInstanceignoreInEval'       ,

    'color'       , # The color of this label
    ])

Mapillary_labels = [
            #name                    clsId  trainId hasInstanceignoreInEval  color
    Label("Bird"                   ,    0,     0,   True,    [165, 42, 42]),
    Label("Ground Animal"          ,    1,     1,   True,    [0, 192, 0]),
    Label("Curb"                   ,    2,     2,   False,    [196, 196, 196]),
    Label("Fence"                  ,    3,     3,   False,    [190, 153, 153]),
    Label("Guard Rail"             ,    4,     4,   False,    [180, 165, 180]),
    Label("Barrier"                ,    5,     5,   False,    [90, 120, 150]),
    Label("Wall"                   ,    6,     6,   False,    [102, 102, 156]),
    Label("Bike Lane"              ,    7,     7,   False,    [128, 64, 255]),
    Label("Crosswalk - Plain"      ,    8,     8,   True,    [140, 140, 200]),
    Label("Curb Cut"               ,    9,     9,   False,    [170, 170, 170]),
    Label("Parking"                ,   10,    10,   False,    [250, 170, 160]),
    Label("Pedestrian Area"        ,   11,    11,   False,    [96, 96, 96]),
    Label("Rail Track"             ,   12,    12,   False,    [230, 150, 140]),
    Label("Road"                   ,   13,    13,   False,    [128, 64, 128]),
    Label("Service Lane"           ,   14,    14,   False,    [110, 110, 110]),
    Label("Sidewalk"               ,   15,    15,   False,    [244, 35, 232]),
    Label("Bridge"                 ,   16,    16,   False,    [150, 100, 100]),
    Label("Building"               ,   17,    17,   False,    [70, 70, 70]),
    Label("Tunnel"                 ,   18,    18,   False,    [150, 120, 90]),
    Label("Person"                 ,   19,    19,   True,    [220, 20, 60]),
    Label("Bicyclist"              ,   20,    20,   True,    [255, 0, 0]),
    Label("Motorcyclist"           ,   21,    21,   True,    [255, 0, 100]),
    Label("Other Rider"            ,   22,    22,   True,    [255, 0, 200]),
    Label("Lane Marking - Crosswalk",   23,    23,   True,    [200, 128, 128]),
    Label("Lane Marking - General" ,   24,    24,   False,    [255, 255, 255]),
    Label("Mountain"               ,   25,    25,   False,    [64, 170, 64]),
    Label("Sand"                   ,   26,    26,   False,    [230, 160, 50]),
    Label("Sky"                    ,   27,    27,   False,    [70, 130, 180]),
    Label("Snow"                   ,   28,    28,   False,    [190, 255, 255]),
    Label("Terrain"                ,   29,    29,   False,    [152, 251, 152]),
    Label("Vegetation"             ,   30,    30,   False,    [107, 142, 35]),
    Label("Water"                  ,   31,    31,   False,    [0, 170, 30]),
    Label("Banner"                 ,   32,    32,   True,    [255, 255, 128]),
    Label("Bench"                  ,   33,    33,   True,    [250, 0, 30]),
    Label("Bike Rack"              ,   34,    34,   True,    [100, 140, 180]),
    Label("Billboard"              ,   35,    35,   True,    [220, 220, 220]),
    Label("Catch Basin"            ,   36,    36,   True,    [220, 128, 128]),
    Label("CCTV Camera"            ,   37,    37,   True,    [222, 40, 40]),
    Label("Fire Hydrant"           ,   38,    38,   True,    [100, 170, 30]),
    Label("Junction Box"           ,   39,    39,   True,    [40, 40, 40]),
    Label("Mailbox"                ,   40,    40,   True,    [33, 33, 33]),
    Label("Manhole"                ,   41,    41,   True,    [100, 128, 160]),
    Label("Phone Booth"            ,   42,    42,   True,    [142, 0, 0]),
    Label("Pothole"                ,   43,    43,   False,    [70, 100, 150]),
    Label("Street Light"           ,   44,    44,   True,    [210, 170, 100]),
    Label("Pole"                   ,   45,    45,   True,    [153, 153, 153]),
    Label("Traffic Sign Frame"     ,   46,    46,   True,    [128, 128, 128]),
    Label("Utility Pole"           ,   47,    47,   True,    [0, 0, 80]),
    Label("Traffic Light"          ,   48,    48,   True,    [250, 170, 30]),
    Label("Traffic Sign (Back)"    ,   49,    49,   True,    [192, 192, 192]),
    Label("Traffic Sign (Front)"   ,   50,    50,   True,    [220, 220, 0]),
    Label("Trash Can"              ,   51,    51,   True,    [140, 140, 20]),
    Label("Bicycle"                ,   52,    52,   True,    [119, 11, 32]),
    Label("Boat"                   ,   53,    53,   True,    [150, 0, 255]),
    Label("Bus"                    ,   54,    54,   True,    [0, 60, 100]),
    Label("Car"                    ,   55,    55,   True,    [0, 0, 142]),
    Label("Caravan"                ,   56,    56,   True,    [0, 0, 90]),
    Label("Motorcycle"             ,   57,    57,   True,    [0, 0, 230]),
    Label("On Rails"               ,   58,    58,   False,    [0, 80, 100]),
    Label("Other Vehicle"          ,   59,    59,   True,    [128, 64, 64]),
    Label("Trailer"                ,   60,    60,   True,    [0, 0, 110]),
    Label("Truck"                  ,   61,    61,   True,    [0, 0, 70]),
    Label("Wheeled Slow"           ,   62,    62,   True,    [0, 0, 192]),
    Label("Car Mount"              ,   63,    63,   False,    [32, 32, 32]),
    Label("Ego Vehicle"            ,   64,    64,   False,    [120, 10, 10]),
    Label("Unlabeled"              ,   65,    65,   False,    [0, 0, 0])]






def draw_image(img):
    img_height, img_width = img.shape

    img_color = np.zeros((img_height, img_width, 3))

    for label in Mapillary_labels:
        if label.hasInstanceignoreInEval:
            
            color = label.color
            trainId = label.trainId
            unq_idx = np.unique(img)
            if trainId not in [54, 55]:
                #print("{} ==> {}".format(label.name, trainId))
                img_color[img == trainId] = color

        if label.trainId in [6, 17] :

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
        vis = np.copy(img)
        img = cv2.cvtColor(vis,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 416))
        img = img/255.
        img = video_tfs(img).unsqueeze(0).to(device, dtype=torch.float)
        pred_raw = model(img)
        pred_y = pred_raw.max(1)[1].cpu().data[0].numpy()
        

        pred_y = pred_y.astype(np.int8)
       #  seg_pred = SegmentationMapsOnImage(pred_y, shape=(416, 512, 3))

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

        # if 0xFF & cv2.waitKey(5) == 27:
        #     break

    

    assert(0)

def GetModel(arch_type="DeepLab", ckpt_path = None, num_cls=None, mode=1):
    print(ckpt_path)
    ckpt = torch.load(ckpt_path)
    model = torch.load(ckpt_path)
    return model
    #print(ckpt.keys())
    print("weight's epoch: {}".format(ckpt["epoch"]))
    if arch_type == "Unet":
        model = Resnet34Unet(3, num_cls)
        model.load_state_dict(ckpt['mode l_state_dict'])
        return model

    elif arch_type == "DeepLab":
        model = DeeplabV3("eval", "./", num_cls, mode=mode)
        model.load_state_dict(ckpt['model_state_dict'])

        return model


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":

    img_h = 416 
    img_w = 512

                               
   
    video_tfs = transforms.Compose([
                                    #transforms.Resize((new_img_h,new_img_w)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                    std=(0.229, 0.224, 0.225))
                        ])

    device = "cuda"
    seg_cls = 65 
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
    ckpt_path = './training_logs/{}/checkpoints/best_valid_serial.pth'.format(folder)
    model = GetModel(arch_type = "DeepLab",
                    ckpt_path = ckpt_path,
                    num_cls = seg_cls,
                    mode = 4)
    
    model.to(device)
    model.eval()

    #model_w = ModelWrapper(model)
    #x = torch.ones((1, 3, img_h, img_w)).cuda()
    #model_trt = torch2trt.torch2trt(model, [x])

    if not os.path.exists("./trt_ep30.pth"):
        torch.save(model_trt.state_dict(), "./trt_ep30.pth")
    from torch2trt import TRTModule

    #model_trt = TRTModule()
    #model_trt.load_state_dict(torch.load("./trt_ep30.pth"))
    video_path = "./exp/20200806_experiment1.avi"
    save_path = './{}_trt_m1_deeplab.avi'.format(os.path.splitext(video_path)[0].split('/')[-1])
    # infer_video(model, video_tfs, device, video_path, save_path)
    cam = cv2.VideoCapture(0)
    while True:
        start_time = time.time()

        ret, img = cam.read()       
        
        img = cv2.resize(img, (img_w, img_h))
        vis = np.copy(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img/255.
        img = video_tfs(img).unsqueeze(0).to(device, dtype=torch.float)
        pred_raw = model(img)
        

        #pred_raw = model_trt(img)
        end_time = time.time()
        
        pred_y = pred_raw.max(1)[1].cpu().data[0].numpy()
        pred_y = pred_y.astype(np.int8)
        pred_img = draw_image(pred_y)
        #cv2.imshow("pred", pred_img)
        ## writer.write(pred_img.astype(np.uint8))
        #resize_vis = cv2.resize(vis, (224, 224))
        overlayed_img = 0.35*vis + 0.65*pred_img
        overlayed_img = overlayed_img.astype(np.uint8)
        #cv2.imshow("overlay_img", overlayed_img)

        ## cv2.imshow("eval", pred_img)
        ## cv2.imwrite("./test/{}.png".format(count), pred_img)
        ## cv2.imwrite("./test/{}_original.png".format(count), resize_vis)
        ## cv2.imwrite("./test/{}_overlayed_img.png".format(count), overlayed_img)
        #assert img_h == overlayed_img.shape[0] and img_w == overlayed_img.shape[1], "wrong H, W"

        combined_img = np.zeros((2*img_h, 2*img_w, 3), dtype=np.uint8)
        combined_img[0:img_h, 0:img_w] = vis
        combined_img[0:img_h, img_w:(2*img_w)] = pred_img
        combined_img[img_h:(2*img_h), (int(img_w/2)):(img_w + int(img_w/2))] = overlayed_img

        fps = 1 / (end_time - start_time)
        fps = str(int(fps))
        cv2.putText(combined_img, "FPS: "+fps, (7, combined_img.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3, cv2.LINE_AA)

        cv2.imshow("frame", combined_img)
        ##out.write(combined_img)

        if 0xFF & cv2.waitKey(5) == 27:
            break



       
