import torch
import torchvision
import random
import logging
from torchvision.transforms import functional as F
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO

def PIL2array1C(img):
    '''Converts a PIL image to NumPy Array

    Args:
        img(PIL Image): Input PIL image
    Returns:
        NumPy Array: Converted image
    '''
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0])


def PIL2array1C(img):
    '''Converts a PIL image to NumPy Array

    Args:
        img(PIL Image): Input PIL image
    Returns:
        NumPy Array: Converted image
    '''
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0],3)

# modified from torchvision to add support for max size
def get_size(min_size, max_size, image_size):
    w, h = image_size
    size = random.choice(min_size)
    max_size = max_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))
    if (w <= h and w == size) or (h <= w and h == size):
        return (h, w)
    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)
    return (oh, ow)

def Resize(min_size, max_size, image):
    size = get_size(min_size, max_size, image.size)
    image = F.resize(image, size)
    return image

def ToTensor(image):
    return F.to_tensor(image)

def Normalize(image, mean, std, to_bgr255=True):
    if to_bgr255:
        image = image[[2, 1, 0]] * 255
    image = F.normalize(image, mean=mean, std=std)
    return image

def is_proc(coco, img_id, ann_info, img):

    MIN_SIZE_TRAIN = (800,)
    MIN_SIZE_RANGE_TRAIN = (-1, -1)
    MAX_SIZE_TRAIN = 1333
    PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
    PIXEL_STD = [1., 1., 1.]
    TO_BGR255 = True
    fg_img_ids = img_id
    bg_img_ids = img_id
    #Transform the input image
    if MIN_SIZE_RANGE_TRAIN[0] == -1:
        min_size = MIN_SIZE_TRAIN
    else:
        assert len(MIN_SIZE_RANGE_TRAIN) == 2, \
            "MIN_SIZE_RANGE_TRAIN must have two elements (lower bound, upper bound)"
        min_size = list(range(
            MIN_SIZE_RANGE_TRAIN[0],
            MIN_SIZE_RANGE_TRAIN[1] + 1
        ))
    to_bgr255 = TO_BGR255
    bg_image_list = []
    root = '/share/Dataset/MSCOCO/images/train2017/'
    bg_path = coco.loadImgs(bg_img_id)[0]['file_name']
    is_number = 0.
    is_en = 0
    is_img_number = [0.,0.,0.,0.,0.,0.,0.,0.]
    is_img_id = 0

    for fg_img_id in fg_img_ids:
        if fg_img_id == bg_img_id:
            bg_path = coco.loadImgs(bg_img_id)[0]['file_name']
            bg_image = Image.open(os.path.join(root, bg_path)).convert('RGB')
            bg_image = Resize(min_size, max_size, bg_image)
            bg_image = ToTensor(bg_image)
            bg_image = Normalize(image=bg_image, mean=PIXEL_MEAN, std=PIXEL_STD, to_bgr255=to_bgr255)
            bg_image_list.append(bg_image) # store the synthetic image
            continue

        fg_path = coco.loadImgs(fg_img_id)[0]['file_name']
        fg_img = Image.open(os.path.join(root, fg_path)).convert('RGB')
        bg_path = coco.loadImgs(bg_img_id)[0]['file_name']
        bg_image = Image.open(os.path.join(root, bg_path)).convert('RGB')
        fg_img_instance = Image.open(os.path.join(root, fg_path)).convert('RGB')
        if bg_image.format != fg_img_instance.format: continue
        bg_ann_ids = coco.getAnnIds(imgIds=bg_img_id)
        bg_ann_ids = list(bg_ann_ids)
        random.shuffle(bg_ann_ids)
        fg_annid_list = []

        for bg_ann_id in bg_ann_ids:
            bg_anno = coco.loadAnns(bg_ann_id)[0]
            if bg_anno["iscrowd"] == 1 : continue
            fg_ann_ids = coco.getAnnIds(imgIds=fg_img_id)
            fg_ann_ids = list(fg_ann_ids)
            random.shuffle(fg_ann_ids)
            for fg_ann_id in fg_ann_ids:
                fg_anno = coco.loadAnns(fg_ann_id)[0]
                if fg_anno["iscrowd"] ==1 : continue
                if fg_anno["category_id"] != bg_anno["category_id"] : continue
                #if fg_ann_id in fg_annid_list: continue
                is_en = 1
                is_img_number[is_img_id] += 1
                bg_xmin, bg_ymin, bg_w, bg_h = bg_anno['bbox']
                fg_xmin, fg_ymin, fg_w, fg_h = fg_anno['bbox']
                bg_xmin = int(round(bg_xmin))
                bg_ymin = int(round(bg_ymin))
                fg_xmin = int(round(fg_xmin))
                fg_ymin = int(round(fg_ymin))
                bg_xmax = bg_xmin + int(round(bg_w))
                bg_ymax = bg_ymin + int(round(bg_h))
                fg_xmax = fg_xmin + int(round(fg_w))
                fg_ymax = fg_ymin + int(round(fg_h))
                fg_mask = coco.annToMask(fg_anno)

                fg_mask_img = Image.fromarray(fg_mask.astype('uint8')).convert('P')
                fg_mask_img = fg_mask_img.crop((fg_xmin, fg_ymin, fg_xmax, fg_ymax))
                fg_mask_img = fg_mask_img.resize(((bg_xmax-bg_xmin),(bg_ymax-bg_ymin)),Image.ANTIALIAS)
                fg_mask_img = Image.fromarray(255-PIL2array1C(fg_mask_img))
                fg_img_instance = fg_img_instance.crop((fg_xmin,fg_ymin,fg_xmax,fg_ymax))
                fg_img_instance = fg_img_instance.resize(((bg_xmax-bg_xmin),(bg_ymax-bg_ymin)),Image.ANTIALIAS)

                bg_image.paste(fg_img_instance,(bg_xmin,bg_ymin),Image.fromarray(cv2.GaussianBlur(PIL2array1C(fg_mask_img),(3,3),2)))
                #plt.imshow(bg_image)
                #plt.show()
                #fg_annid_list.append(fg_ann_id)
                break

        bg_image = Resize(min_size, max_size, bg_image)
        bg_image = ToTensor(bg_image)
        bg_image = Normalize(image=bg_image, mean=PIXEL_MEAN, std=PIXEL_STD, to_bgr255=to_bgr255)
        bg_image = PIL2array3C(bg_img)
        bg_image_list.append(bg_image) # store the synthetic image or the original image
        is_img_id+=1
        if is_en == 1:
            is_number+=1
            is_en=0
        
    return bg_image_list


