import math
import os
import random

import cv2
import numpy
import torch
from PIL import Image
from torch.utils import data
import glob
import albumentations
from utils import util
import json

class VPDataset(data.Dataset):
    def __init__(self, folder, input_size, params, augment):
        self.params = params
        self.mosaic = augment
        self.augment = augment
        self.input_size = input_size
        
        self.albumentations = Albumentations()
        self.vp_loader = LoadVisualPrompt()
        self.data = self.read_data(folder)
        
    def read_data(self, folder):
        data = []
        videos = [entry.name for entry in os.scandir(folder) if entry.is_dir()]
        for video in videos:
            boxes = {}
            video_path = os.path.join(folder, video)
            imgs = glob.glob(f"{video_path}/*.jpg")
            with open(f"{video_path}/groundtruth.txt", "r") as file:
                lines = file.readlines()
            for img in imgs:
                i = int(img.split("/")[-1].split(".")[0]) - 1
                boxes[img] = [-1] + [float(v) for v in lines[i].strip().split(",")]
            
            prompt_imgs = self.top_k_boxes(boxes)
            for img, box in boxes.items():
                if img not in prompt_imgs:
                    prompt_img = random.choice(prompt_imgs)
                    data.append({
                        "img": img,
                        "box": numpy.array([box]),
                        "prompt_img": prompt_img,
                        "prompt_box": numpy.array([boxes[prompt_img]])
                    })
        return data
            
    def top_k_boxes(self, boxes, k=4):
        n = len(boxes)
        k = max(1, min(int(0.4 * n), k))

        areas = {
            img: w * h
            for img, (_, x, y, w, h) in boxes.items()
        }
        sorted_boxes = sorted(areas.items(), key=lambda x: x[1], reverse=True)
        return [x[0] for x in sorted_boxes[:k]]

    def __getitem__(self, index):
        item = self.data[index]
        img, shape = self.load_image(item["img"], self.augment)
        query_img,  query_box = self.process_image(img, shape, item["box"], self.augment)
        img, shape = self.load_image(item["prompt_img"])
        prompt_img, prompt_box = self.process_image(img, shape, item["prompt_box"])
        prompt_mask = self.vp_loader(prompt_img, prompt_box)
        return query_img,  query_box, prompt_img, prompt_mask
    
    def process_image(self, image, shape, item_box, augment= False):
        h, w = image.shape[:2]

        # Resize
        image, ratio, pad = resize(image, self.input_size, augment)

        label = item_box.copy()
        label[:,1:] = coco2wh(label[:, 1:], shape[1], shape[0])
        
        if label.size:
            label[:, 1:] = wh2xy(label[:, 1:], ratio[0] * w, ratio[1] * h, pad[0], pad[1])
        if augment:
            new_image, new_label = random_perspective(image.copy(), label.copy(), self.params)
            if len(new_label) > 0:
                image, label = new_image, new_label

        h, w = image.shape[:2]
        box = label[:, 1:5]
        box = xy2wh(box, w, h)

        if augment:
            # Albumentations
            image, box = self.albumentations(image, box)
            # HSV color-space
            augment_hsv(image, self.params)
            # Flip up-down
            if random.random() < self.params['flip_ud']:
                image = numpy.flipud(image)
                box[:, 1] = 1 - box[:, 1]
            # Flip left-right
            if random.random() < self.params['flip_lr']:
                image = numpy.fliplr(image)
                box[:, 0] = 1 - box[:, 0]

        target_box = torch.from_numpy(box)

        # Convert HWC to CHW, BGR to RGB
        sample = image.transpose((2, 0, 1))[::-1]
        sample = numpy.ascontiguousarray(sample)

        return torch.from_numpy(sample), target_box
    def __len__(self):
      return len(self.data)
    
    def load_image(self, filename, augment=False):
        image = cv2.imread(filename)
        h, w = image.shape[:2]
        r = self.input_size / max(h, w)
        if r != 1:
            image = cv2.resize(image,
                               dsize=(int(w * r), int(h * r)),
                               interpolation=resample() if augment else cv2.INTER_LINEAR)
        return image, (h, w)
    
    @staticmethod
    def collate_fn(batch):
        return []
    
class ZaloVPDataset(VPDataset):
    def __init__(self, folder, input_size, params, augment):
        super().__init__(folder, input_size, params, augment)
    
    def read_data(self, folder):
        data = []
        videos = [entry.name for entry in os.scandir(folder) if entry.is_dir()]

        with open(f"{folder}/data.json") as f:
            frame_data = json.load(f)
            
        with open(f"{folder}/prompt_data.json") as f:
            prompt_data = {}
            for k, v in json.load(f).items():
                nk = "/".join([k.split("/")[0][:-2]] + k.split("/")[1:])
                prompt_data[nk] = v
        
        for video in videos:
            video_path = os.path.join(folder, video)
            imgs = glob.glob(f"{video_path}/*.jpg")
            prompt = glob.glob(f"{video_path}/prompt/*.jpg")
            
            for img in imgs:
                box_key =  "/".join(img.split("/")[-2:])
                [x1, y1, x2, y2] = frame_data[box_key]
                box = [-1] + [x1, y1, x2 - x1, y2- y1]

                prompt_img = random.choice(prompt)
                prompt_key =  "/".join([prompt_img.split("/")[-3][:-2]] + prompt_img.split("/")[-2:])
                [x1, y1, x2, y2] = prompt_data[prompt_key]
                prompt_box = [-1] + [x1, y1, x2 - x1, y2- y1]

                data.append({
                            "img": img,
                            "box": numpy.array([box], dtype=float),
                            "prompt_img": prompt_img,
                            "prompt_box": numpy.array([prompt_box], dtype=float)
                            })
                    
        return data
    
    def __getitem__(self, index):
        return super().__getitem__(index)

class LoadVisualPrompt:
    def __init__(self):
        self.scale_factor = 1/8
    
    def make_mask(self, boxes, h, w):
        x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
        r = torch.arange(w)[None, None, :]  # rows shape(1,1,w)
        c = torch.arange(h)[None, :, None]  # cols shape(1,h,1)

        return ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))
    
    def __call__(self, image, target_box):
        imgsz = image.shape[1:]
        masksz = (int(imgsz[0] * self.scale_factor), int(imgsz[1] * self.scale_factor))

        box = util.xywh2xyxy(target_box) * torch.tensor(masksz)[[1, 0, 1, 0]]  # target boxes
        masks = self.make_mask(box, *masksz).float()
        
        return masks
    
    
def coco2wh(x, w=640, h=640):
    """
    Convert nx4 boxes from [xmin, ymin, w, h] (COCO)
    to [x_center, y_center, w, h] normalized format.
    """
    y = numpy.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2] / 2 ) / w  # x_center normalized
    y[:, 1] = (x[:, 1] + x[:, 3] / 2 ) / h  # y_center normalized
    y[:, 2] = x[:, 2] / w                          # width normalized
    y[:, 3] = x[:, 3] / h                          # height normalized
    return y


def wh2xy(x, w=640, h=640, pad_w=0, pad_h=0):
    # Convert nx4 boxes
    # from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = numpy.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + pad_w  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + pad_h  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + pad_w  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + pad_h  # bottom right y
    return y


def xy2wh(x, w, h):
    # warning: inplace clip
    x[:, [0, 2]] = x[:, [0, 2]].clip(0, w - 1E-3)  # x1, x2
    x[:, [1, 3]] = x[:, [1, 3]].clip(0, h - 1E-3)  # y1, y2

    # Convert nx4 boxes
    # from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    y = numpy.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def resample():
    choices = (cv2.INTER_AREA,
               cv2.INTER_CUBIC,
               cv2.INTER_LINEAR,
               cv2.INTER_NEAREST,
               cv2.INTER_LANCZOS4)
    return random.choice(seq=choices)


def augment_hsv(image, params):
    # HSV color-space augmentation
    h = params['hsv_h']
    s = params['hsv_s']
    v = params['hsv_v']

    r = numpy.random.uniform(-1, 1, 3) * [h, s, v] + 1
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

    x = numpy.arange(0, 256, dtype=r.dtype)
    lut_h = ((x * r[0]) % 180).astype('uint8')
    lut_s = numpy.clip(x * r[1], 0, 255).astype('uint8')
    lut_v = numpy.clip(x * r[2], 0, 255).astype('uint8')

    hsv = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v)))
    cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR, dst=image)  # no return needed


def resize(image, input_size, augment):
    # Resize and pad image while meeting stride-multiple constraints
    shape = image.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(input_size / shape[0], input_size / shape[1])
    if not augment:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    pad = int(round(shape[1] * r)), int(round(shape[0] * r))
    w = (input_size - pad[0]) / 2
    h = (input_size - pad[1]) / 2

    if shape[::-1] != pad:  # resize
        image = cv2.resize(image,
                           dsize=pad,
                           interpolation=resample() if augment else cv2.INTER_LINEAR)
    top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
    left, right = int(round(w - 0.1)), int(round(w + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)  # add border
    return image, (r, r), (w, h)


def candidates(box1, box2):
    # box1(4,n), box2(4,n)
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    aspect_ratio = numpy.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > 2) & (h2 > 2) & (w2 * h2 / (w1 * h1 + 1e-16) > 0.1) & (aspect_ratio < 100)


def random_perspective(image, label, params, border=(0, 0)):
    h = image.shape[0] + border[0] * 2
    w = image.shape[1] + border[1] * 2

    # Center
    center = numpy.eye(3)
    center[0, 2] = -image.shape[1] / 2  # x translation (pixels)
    center[1, 2] = -image.shape[0] / 2  # y translation (pixels)

    # Perspective
    perspective = numpy.eye(3)

    # Rotation and Scale
    rotate = numpy.eye(3)
    a = random.uniform(-params['degrees'], params['degrees'])
    s = random.uniform(1 - params['scale'], 1 + params['scale'])
    rotate[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    shear = numpy.eye(3)
    shear[0, 1] = math.tan(random.uniform(-params['shear'], params['shear']) * math.pi / 180)
    shear[1, 0] = math.tan(random.uniform(-params['shear'], params['shear']) * math.pi / 180)

    # Translation
    translate = numpy.eye(3)
    translate[0, 2] = random.uniform(0.5 - params['translate'], 0.5 + params['translate']) * w
    translate[1, 2] = random.uniform(0.5 - params['translate'], 0.5 + params['translate']) * h

    # Combined rotation matrix, order of operations (right to left) is IMPORTANT
    matrix = translate @ shear @ rotate @ perspective @ center
    if (border[0] != 0) or (border[1] != 0) or (matrix != numpy.eye(3)).any():  # image changed
        image = cv2.warpAffine(image, matrix[:2], dsize=(w, h), borderValue=(0, 0, 0))

    # Transform label coordinates
    n = len(label)
    if n:
        xy = numpy.ones((n * 4, 3))
        xy[:, :2] = label[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ matrix.T  # transform
        xy = xy[:, :2].reshape(n, 8)  # perspective rescale or affine

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        box = numpy.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip
        box[:, [0, 2]] = box[:, [0, 2]].clip(0, w)
        box[:, [1, 3]] = box[:, [1, 3]].clip(0, h)
        # filter candidates
        indices = candidates(box1=label[:, 1:5].T * s, box2=box.T)

        label = label[indices]
        label[:, 1:5] = box[indices]

    return image, label


def mix_up(image1, label1, image2, label2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    alpha = numpy.random.beta(a=32.0, b=32.0)  # mix-up ratio, alpha=beta=32.0
    image = (image1 * alpha + image2 * (1 - alpha)).astype(numpy.uint8)
    label = numpy.concatenate((label1, label2), 0)
    return image, label


class Albumentations:
    def __init__(self):
        self.transform = None
        try:

            transforms = [albumentations.Blur(p=0.01),
                          albumentations.CLAHE(p=0.01),
                          albumentations.ToGray(p=0.01),
                          albumentations.MedianBlur(p=0.01)]
            self.transform = albumentations.Compose(transforms)

        except ImportError:  # package not installed, skip
            pass

    def __call__(self, image, box):
        if self.transform:
            x = self.transform(image=image,
                               bboxes=box)
            image = x['image']
            box = numpy.array(x['bboxes'])
        return image, box