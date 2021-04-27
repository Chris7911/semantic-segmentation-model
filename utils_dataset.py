from collections import namedtuple
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
Label = namedtuple('Label', [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class
    'clsId'       ,

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

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ])

apollo_labels = [
    #     name                    clsId    id   trainId   category  catId  hasInstanceignoreInEval   color
    Label('others'              ,    0 ,    0,    0   , '其他'    ,   0  ,False , True  , 0x000000 ),
    Label('rover'               , 0x01 ,    1,    0   , '其他'    ,   0  ,False , True  , 0x000000 ),
    Label('sky'                 , 0x11 ,   17,    0    , '天空'    ,   1  ,False , False , 0x000000 ),
    Label('car'                 , 0x21 ,   33,    1    , '移动物体',   2  ,True  , False , 0x00008E ),
    Label('car_groups'          , 0xA1 ,  161,    1    , '移动物体',   2  ,True  , False , 0x00008E ),
    Label('motorbicycle'        , 0x22 ,   34,    2    , '移动物体',   2  ,True  , False , 0x0000E6 ),
    Label('motorbicycle_group'  , 0xA2 ,  162,    2    , '移动物体',   2  ,True  , False , 0x0000E6 ),
    Label('bicycle'             , 0x23 ,   35,    3    , '移动物体',   2  ,True  , False , 0x770B20 ),
    Label('bicycle_group'       , 0xA3 ,  163,    3    , '移动物体',   2  ,True  , False , 0x770B20 ),
    Label('person'              , 0x24 ,   36,    4    , '移动物体',   2  ,True  , False , 0x0080c0 ),
    Label('person_group'        , 0xA4 ,  164,    4    , '移动物体',   2  ,True  , False , 0x0080c0 ),
    Label('rider'               , 0x25 ,   37,    5    , '移动物体',   2  ,True  , False , 0x804080 ),
    Label('rider_group'         , 0xA5 ,  165,    5    , '移动物体',   2  ,True  , False , 0x804080 ),
    Label('truck'               , 0x26 ,   38,    6    , '移动物体',   2  ,True  , False , 0x8000c0 ),
    Label('truck_group'         , 0xA6 ,  166,    6    , '移动物体',   2  ,True  , False , 0x8000c0 ),
    Label('bus'                 , 0x27 ,   39,    7    , '移动物体',   2  ,True  , False , 0xc00040 ),
    Label('bus_group'           , 0xA7 ,  167,    7    , '移动物体',   2  ,True  , False , 0xc00040 ),
    Label('tricycle'            , 0x28 ,   40,    8    , '移动物体',   2  ,True  , False , 0x8080c0 ),
    Label('tricycle_group'      , 0xA8 ,  168,    8    , '移动物体',   2  ,True  , False , 0x8080c0 ),
    Label('road'                , 0x31 ,   49,    9    , '平面'    ,   3  ,False , False , 0xc080c0 ),
    Label('siderwalk'           , 0x32 ,   50,    10   , '平面'    ,   3  ,False , False , 0xc08040 ),
    Label('traffic_cone'        , 0x41 ,   65,    11   , '路间障碍',   4  ,False , False , 0xff8040 ),
    Label('road_pile'           , 0x42 ,   66,    12   , '路间障碍',   4  ,False , False , 0x0000c0 ),
    Label('fence'               , 0x43 ,   67,    13   , '路间障碍',   4  ,False , False , 0x404080 ),
    Label('traffic_light'       , 0x51 ,   81,    14   , '路边物体',   5  ,False , False , 0xc04080 ),
    Label('pole'                , 0x52 ,   82,    15   , '路边物体',   5  ,False , False , 0xc08080 ),
    Label('traffic_sign'        , 0x53 ,   83,    16   , '路边物体',   5  ,False , False , 0x004040 ),
    Label('wall'                , 0x54 ,   84,    17   , '路边物体',   5  ,False , False , 0xc0c080 ),
    Label('dustbin'             , 0x55 ,   85,    18   , '路边物体',   5  ,False , False , 0x4000c0 ),
    Label('billboard'           , 0x56 ,   86,    19   , '路边物体',   5  ,False , False , 0xc000c0 ),
    Label('building'            , 0x61 ,   97,    20   , '建筑'    ,   6  ,False , False , 0xc00080 ),
    Label('bridge'              , 0x62 ,   98,    21  , '建筑'    ,   6  ,False , True  , 0x808000 ),
    Label('tunnel'              , 0x63 ,   99,    22  , '建筑'    ,   6  ,False , True  , 0x800000 ),
    Label('overpass'            , 0x64 ,  100,    23  , '建筑'    ,   6  ,False , True  , 0x408040 ),
    Label('vegatation'          , 0x71 ,  113,    24   , '自然'    ,   7  ,False , False , 0x808040 ),
    Label('unlabeled'           , 0xFF ,  255,    0  , '未标注'  ,   8  ,False , True  , 0x000000 ),
]

# name to label object
name2label      = {label.name: label for label in apollo_labels}

# id to label object
id2label        = {label.id: label for label in apollo_labels}

# trainId to label object
trainId2label   = {label.trainId: label for label in reversed(apollo_labels)}

# category to list of label objects
category2labels = {}
for label in apollo_labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]

color2label = {}
for label in apollo_labels:
    #color = (int(label.color[2:4],16),int(label.color[4:6],16),int(label.color[6:8],16))
    color = label.color
    r =  color // (256*256)
    g = (color-256*256*r) // 256
    b = (color-256*256*r-256*g)
    color2label[(r, g, b)] = label


def assureSingleInstanceName(name):
    # if the name is known, it is not a group
    if name in name2label:
        return name
    # test if the name actually denotes a group
    if not name.endswith("group"):
        return None
    # remove group
    name = name[:-len("group")]
    # test if the new name exists
    if not name in name2label:
        return None
    # test if the new name denotes a label that actually has instances
    if not name2label[name].hasInstances:
        return None
    # all good then
    return name

# print("")
# print("    {:>21} | {:>3} | {:>7} | {:>14} |".format('name', 'id', 'trainId', 'category')\
# +  "{:>10} | {:>12} | {:>12}".format('categoryId', 'hasInstances', 'ignoreInEval'))
# print("    " + ('-' * 98))
# for label in apollo_labels:
#     print("    {:>21} | {:>3} | {:>7} |".format(label.name, label.id, label.trainId)\
#     + "  {:>14} |{:>10} ".format(label.category, label.categoryId)\
#     + "| {:>12} | {:>12}".format(label.hasInstances, label.ignoreInEval ))
# print("")

def SHOW_IMAGE_LABEL():
    import numpy as np
    from PIL import Image
    from imgaug import augmenters as iaa
    import imgaug
    import cv2

    with open("./image_train.txt", "r") as fp:
        imgs_train =[f.strip('\n') for f in fp.readlines()]
    
    with open("./label_train.txt", "r") as fp:
        masks_train =[f.strip('\n') for f in fp.readlines()]

    with open("./image_valid.txt", "r") as fp:
        imgs_valid =[f.strip('\n') for f in fp.readlines()]
    
    with open("./label_valid.txt", "r") as fp:
        masks_valid =[f.strip('\n') for f in fp.readlines()]

    for img_p, label_p in zip(imgs_train, masks_train):

        im = Image.open(img_p)
        label = Image.open(label_p)


        im_resize = im.resize((512, 416))
        label_resize = label.resize((512, 416))


        image = np.asarray(im_resize, dtype=np.uint8)
        label = np.asarray(label_resize, dtype=np.int32)
        # label = np.resize(label, (416, 512))
        seg = np.copy(label)

        
        for k, v in id2label.items():
            seg[ label == k ] = v.trainId


        # seg= SegmentationMapsOnImage(seg, shape=image.shape)
        # imgaug.imshow(seg.draw()[0])
        # imgaug.imshow(seg.draw_on_image(image, alpha=0.6)[0])



        seg = np.repeat(seg[:, :, np.newaxis], 3, axis=2)
        c = np.ones(seg.shape)
        flag = False
        for k, v in color2label.items():
            for _v in v:
                row, col, _ = np.where(seg == _v.trainId)
                if len(row) and _v.trainId == 0:
                    
                    # if k == (192, 128, 192):
                        # print("===========================")
                    c[row, col, 0] = 255
                    c[row, col, 1] = 0
                    c[row, col, 2] = 0
                    flag = True
        if flag:
            cv2.imshow("label", c)
            cv2.imwrite("./213.png", c)
            cv2.waitKey(0)
            # assert(0)
            

if __name__ == "__main__":
    if False:
        SHOW_IMAGE_LABEL()