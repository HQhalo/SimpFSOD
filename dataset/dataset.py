import os
import random

from dataset.resize import LetterBox
import numpy
import torch
from torch.utils import data
import glob
from utils import util
import json
import cv2

class VPDataset(data.Dataset):
    def __init__(self, folder, input_size, params, augment):
        self.params = params
        self.mosaic = augment
        self.augment = augment
        self.input_size = input_size
        
        self.vp_loader = LoadVisualPrompt()
        self.letter_box = LetterBox(input_size, params)

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

        # query
        query_img = self.load_image(item["img"])
        query_img, query_box = self.letter_box(query_img, item["box"], augment=True, coco_fotmat=True)
        
        # prompt
        prompt_img = self.load_image(item["prompt_img"])
        prompt_img, prompt_box = self.letter_box(prompt_img, item["prompt_box"], augment=False, coco_fotmat=True)
        prompt_mask = self.vp_loader(prompt_img, prompt_box)

        return query_img, query_box , prompt_img, prompt_mask
    
    def load_image(self, filename):
        return cv2.imread(filename)
    
    def __len__(self):
      return len(self.data)
    
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
  