import cv2
import numpy as np
from PIL import Image
import math
import os
import imgaug
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
assert float(cv2.__version__.rsplit('.', 1)[0]) >= 3, 'OpenCV version 3 or newer required.'



# def dist(x, y):
#     return sqrt(x*x + y*y)

# def correct_fisheye(src_size, dest_size, dx, dy, factor):
#     rx, ry = dx-(dest_size[0]/2), dy-(dest_size[1]/2)
#     r = dist(rx, ry) / (dist(src_size[0], src_size[1])/factor)
#     if r == 0:
#         theta = 1.0
#     else:
#         theta = atan(r)/r
#     sx, sy = (src_size[0]/2) + theta * rx, (src_size[1]/2)+theta*ry
#     return (int(round(sx)), int(round(sy)))

# https://hal.inria.fr/inria-00267247/document
# https://stackoverflow.com/questions/45459088/reverse-fisheye-radial-distortion-using-field-of-view-model

def distortFishEye(src, w):
    map_x = np.zeros((src.shape[0], src.shape[1]), dtype=np.float32)
    map_y = np.zeros((src.shape[0], src.shape[1]), dtype=np.float32)
    Cx  = src.shape[1]/2
    Cy = src.shape[0]/2

    for x in np.arange(-1.0, 1.0, 1.0/Cx):
        for y in np.arange(-1.0, 1.0, 1.0/Cy):
            ru = math.sqrt(x*x + y*y)
            rd = (1.0 / w)*math.atan(2.0* ru * math.tan(w / 2.0))
            map_x[int(y*Cy + Cy), int(x*Cx + Cx)] = rd/ru * x*Cx + Cx
            map_y[int(y*Cy + Cy), int(x*Cx + Cx)] = rd/ru * y*Cy + Cy
    dst = cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR)
    return dst

def distortFishEye2(src, w):
    map_x = np.zeros((src.shape[0], src.shape[1]), dtype=np.float32)
    map_y = np.zeros((src.shape[0], src.shape[1]), dtype=np.float32)
    cx  = src.shape[1]/2
    cy = src.shape[0]/2

    for x in np.arange(0, src.shape[1]):
        for y in np.arange(0, src.shape[0]):
            rx = (x - cx) / cx
            ry = (y - cy) / cy
            ru = math.sqrt(rx*rx + ry*ry) + 0.000000001
            # ru = math.sqrt(x*x + y*y)
            rd = (1.0 / w)*math.atan(2.0* ru * math.tan(w / 2.0))
            coeff = rd / ru
            rx *= coeff
            ry *= coeff

            map_x[int(y), int(x)] = rx*cx + cx
            map_y[int(y), int(x)] = ry*cy + cy
    dst = cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR)
    return dst

def undistortFishEye(src, w):
    map_x = np.zeros((src.shape[0], src.shape[1]), dtype=np.float32)
    map_y = np.zeros((src.shape[0], src.shape[1]), dtype=np.float32)
    Cx  = src.shape[1]/2
    Cy = src.shape[0]/2

    for x in np.arange(-1.0, 1.0, 1.0/Cx):
        for y in np.arange(-1.0, 1.0, 1.0/Cy):
            ru = math.sqrt(x*x + y*y)
            rd = (1. / w) * math.atan(2.*ru*math.tan(w/2.))

            map_x[int(y*Cy + Cy), int(x*Cx + Cx)] = rd/ru * x*Cx + Cx
            map_y[int(y*Cy + Cy), int(x*Cx + Cx)] = rd/ru * y*Cy + Cy
    dst = cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR)

    return dst


def func_images(images, random_state, parents, hooks):
    w = (185 / 2*math.pi /180)
    result = []
    for image in images:
        height, width = image.shape[:2]
        mask = np.zeros((height,width, 3), np.uint8)
        x, y, r = int(width/2), int(height/2), int(width*0.5)
        mask = cv2.circle(mask, (x, y), r, (255, 255, 255), -1, 8, 0)
        map_x = np.zeros((height, width), dtype=np.float32)
        map_y = np.zeros((height, width), dtype=np.float32)
        Cx  = width/2
        Cy = height/2

        for x in np.arange(-1.0, 1.0, 1.0/Cx):
            for y in np.arange(-1.0, 1.0, 1.0/Cy):
                ru = math.sqrt(x*x + y*y)
                rd = (1.0 / w)*math.atan(2.0* ru * math.tan(w / 2.0))
                map_x[int(y*Cy + Cy), int(x*Cx + Cx)] = rd/ru * x*Cx + Cx
                map_y[int(y*Cy + Cy), int(x*Cx + Cx)] = rd/ru * y*Cy + Cy
        out = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        out = out * (mask==255)
        result.append(out)
    return result

def func_seg(segmaps, random_state, parents, hooks):
    w = (185 / 2*math.pi /180)
    result = []
    for segmap in segmaps:
        height, width = segmap.get_arr().shape[:2]
        mask = np.zeros((height, width), np.uint8)
        x, y, r = int(width/2), int(height/2), int(width*0.5)
        mask = cv2.circle(mask, (x, y), r, (255, 255, 255), -1, 8, 0)
        out = segmap.get_arr() * (mask==255) + 65 * (mask!=255)
        seg = SegmentationMapsOnImage(out, shape=segmap.shape)
        result.append(seg)
    return result


distort_augmenter = iaa.Lambda(func_images = func_images,
                                func_segmentation_maps = func_seg)

if __name__ == "__main__":
    src = []
    W = (185 / 2*math.pi /180)
    test_p = {"img_path": "D://mapillary-vistas-dataset\\validation\\images\\-4jzRzGfKmQg8RBNlNqnGQ.jpg",
            "label_path": "D://mapillary-vistas-dataset\\validation\\labels\\-4jzRzGfKmQg8RBNlNqnGQ.png",
            "instance_path": "D://mapillary-vistas-dataset\\validation\\instances\\-4jzRzGfKmQg8RBNlNqnGQ.png",
            "panoptic_path": "D://mapillary-vistas-dataset\\validation\\panoptic\\-4jzRzGfKmQg8RBNlNqnGQ.png"}
    src.append(test_p["img_path"])
    src.append(test_p["instance_path"])
    new_img_h = 480
    new_img_w = 512
    seq = iaa.Sequential([iaa.size.Resize({"height": new_img_h, "width": new_img_w}, interpolation='nearest'),
                            iaa.Fliplr(0.5),
                            distort_augmenter])


    # for i, img_p in enumerate(src):
    #     img = cv2.imread(img_p)
    #     height, width = img.shape[:2]
    #     mask = np.zeros((height,width, 3), np.uint8)
    #     x, y, r = int(width/2), int(height/2), int(width*0.5)
    #     mask = cv2.circle(mask,(x, y), r, (255, 255, 255), -1, 8, 0)
    #     dst_u1 = distortFishEye(img, W)
    #     out = dst_u1 * (mask==255)
    #     cv2.imwrite("{}_distort.png".format(os.path.splitext(img_p)[0]), seq_img)
    
    
    a = cv2.imread(test_p["img_path"])

    b = cv2.imread(test_p["instance_path"], -1)
    b = np.array(b/256., dtype=np.uint8)
    # print(np.unique(b))
    seg = SegmentationMapsOnImage(b, shape=a.shape)
    a, b = seq(image=a, segmentation_maps=seg)
    a = np.array(a, dtype=np.uint8)
    b = b.get_arr()
    b = np.array(b, dtype=np.uint8)
    cv2.imshow("a", a)
    cv2.imshow("b", b)
    # print(np.unique(b))
    # print(b[0][0])
    cv2.waitKey(0)



