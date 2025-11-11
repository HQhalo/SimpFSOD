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
                boxes[img] = [0] + [float(v) for v in lines[i].strip().split(",")]
            
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
        query_img, query_box, _ = self.letter_box(query_img, item["box"], augment=self.augment, coco_fotmat=True)
        
        # prompt
        prompt_img = self.load_image(item["prompt_img"])
        prompt_img, prompt_box, _ = self.letter_box(prompt_img, item["prompt_box"], augment=False, coco_fotmat=True)
        prompt_mask = self.vp_loader(prompt_img, prompt_box)

        return query_img, query_box , prompt_img, prompt_mask
    
    def load_image(self, filename):
        return cv2.imread(filename)
    
    def __len__(self):
      return len(self.data)
    
class ZaloVPDataset(VPDataset):
    def __init__(self, folder, input_size, params, augment, videos):
        self.cache = {}
        self.videos = videos
        super().__init__(folder, input_size, params, augment)
    
    def read_data(self, folder):
        data = []
        with open(f"{folder}/data.json") as f:
            frame_data = json.load(f)
            
        with open(f"{folder}/prompt_data.json") as f:
            prompt_data = {}
            for k, v in json.load(f).items():
                nk = "/".join([k.split("/")[0][:-2]] + k.split("/")[1:])
                prompt_data[nk] = v
        
        for video in self.videos:
            video_path = os.path.join(folder, video)
            imgs = glob.glob(f"{video_path}/*.jpg")
            prompt = glob.glob(f"{video_path}/prompt/*.jpg")
            
            for img in imgs:
                box_key =  "/".join(img.split("/")[-2:])
                [x1, y1, x2, y2] = frame_data[box_key]
                box = [0, x1, y1, x2 - x1, y2- y1]

                prompt_img = random.choice(prompt)
                prompt_key =  "/".join([prompt_img.split("/")[-3][:-2]] + prompt_img.split("/")[-2:])
                [x1, y1, x2, y2] = prompt_data[prompt_key]
                prompt_box = [0, x1, y1, x2 - x1, y2- y1]

                data.append({
                            "img": img,
                            "box": numpy.array([box], dtype=float),
                            "prompt_img": prompt_img,
                            "prompt_box": numpy.array([prompt_box], dtype=float)
                            })
                    
        return data
    
    def __getitem__(self, index):
        item = self.data[index]

        # query
        query_img = self.load_image(item["img"])
        query_img, query_box, query_cls = self.letter_box(query_img, item["box"], augment=True, coco_fotmat=True)
        
        # prompt
        if item["prompt_img"] not in self.cache:
            prompt_img = self.load_image(item["prompt_img"])
            prompt_img, prompt_box, prompt_cls = self.letter_box(prompt_img, item["prompt_box"], augment=False, coco_fotmat=True)
            prompt_mask = self.vp_loader(prompt_img, prompt_box, prompt_cls)
            self.cache[item["prompt_img"]] = (prompt_img, prompt_mask)

        (prompt_img, prompt_mask) = self.cache[item["prompt_img"]]            
        return query_img, query_box , query_cls, prompt_img, prompt_mask, torch.zeros(len(item["box"]))
    
    @staticmethod
    def collate_fn(batch):
        samples, box, cls, prompt_img, prompt_mask, indices = zip(*batch)
        
        box = torch.cat(box, dim=0)
        cls = torch.cat(cls, dim=0)
        
        new_indices = list(indices)
        for i in range(len(indices)):
            new_indices[i] += i
        indices = torch.cat(new_indices, dim=0)

        targets = {'box': box,
                   'idx': indices,
                   'cls': cls}
        return torch.stack(samples, dim=0), torch.stack(prompt_img, dim=0), torch.stack(prompt_mask, dim=0), targets
    
class SyntheticVPDataset(VPDataset):
    def __init__(self, folder, input_size, params, augment):
        self.cache = {}
        super().__init__(folder, input_size, params, augment)
    
    def read_data(self, folder):
        data = []
        with open(f"{folder}/annotations/annotations.json") as f:
            frame_data = json.load(f)
        
        videos = [entry.name for entry in os.scandir(f"{folder}/samples") if entry.is_dir()]

        for video in videos:
            video_path = os.path.join(folder, "samples", video)
            imgs = glob.glob(f"{video_path}/images/*.jpg")
            prompt = glob.glob(f"{video_path}/object_images/*.png")
            
            for img in imgs:
                box_key = img.split("/")[-1]
                [x1, y1, w, y] = frame_data[box_key]
                box = [0, x1, y1, w, y]

                prompt_img = random.choice(prompt)
                
                data.append({
                            "img": img,
                            "box": numpy.array([box], dtype=float),
                            "prompt_img": prompt_img
                            })
                    
        return data
    
    def __getitem__(self, index):
        item = self.data[index]

        # query
        query_img = self.load_image(item["img"])
        query_img, query_box, query_cls = self.letter_box(query_img, item["box"], augment=True, coco_fotmat=True)
        
        # prompt
        
        prompt_img = self.load_prompt_image(item["prompt_img"])
        prompt_img, _, prompt_cls, alpha = self.letter_box(prompt_img, numpy.array([[0,1,1,10,10]]), augment=False, coco_fotmat=True)
        alpha = (alpha > 0).float().unsqueeze(0)
        prompt_mask = self.vp_loader(prompt_img, None, prompt_cls, alpha)
                      
        return query_img, query_box , query_cls, prompt_img, prompt_mask, torch.zeros(len(item["box"]))
    
    def load_prompt_image(self, filename):
        return cv2.imread(filename, cv2.IMREAD_UNCHANGED) 
    
    @staticmethod
    def collate_fn(batch):
        samples, box, cls, prompt_img, prompt_mask, indices = zip(*batch)
        
        box = torch.cat(box, dim=0)
        cls = torch.cat(cls, dim=0)
        
        new_indices = list(indices)
        for i in range(len(indices)):
            new_indices[i] += i
        indices = torch.cat(new_indices, dim=0)

        targets = {'box': box,
                   'idx': indices,
                   'cls': cls}
        return torch.stack(samples, dim=0), torch.stack(prompt_img, dim=0), torch.stack(prompt_mask, dim=0), targets
    
class VisDroneDataset(data.Dataset):
    def __init__(self, folder, input_size, params, augment, nc=1):
        self.params = params
        self.mosaic = augment
        self.augment = augment
        self.input_size = input_size
        
        self.vp_loader = LoadVisualPrompt(nc)
        self.letter_box = LetterBox(input_size, params)

        self.data = self.read_data(folder)

    def read_data(self, folder):
        data = []
        annotations_files = glob.glob(f"{folder}/annotations/*.txt")
        annotations_dict = {}
        for annotations_file in annotations_files:
            file_id = annotations_file.split("/")[-1].split(".")[0]
            with open(annotations_file) as file:
                lines = file.readlines()
                lines = [line.strip().split(",") for line in lines]
                boxes = []
                for line in lines:
                    [x,y,w,h,score,cls,truncation,occlusion ] = [int(i) for i in line[:8]]
                    if score == 0 or occlusion == 2 or cls == 11:
                        continue 
                    boxes.append([cls, x,y,w,h]) 
                if len(boxes) != 0:
                    annotations_dict[file_id] = boxes

        for file_id, annotations in annotations_dict.items():
            data.append({
                        "img": f"{folder}/images/{file_id}.jpg",
                        "box": numpy.array(annotations, dtype=float)
                    })
        return data
    def __getitem__(self, index):
        item = self.data[index]
        img = self.load_image(item["img"])
        img, box, cls = self.letter_box(img, item["box"], augment=self.augment, coco_fotmat=True)
        prompt_mask = self.vp_loader(img, box, cls)
        return img, box, cls, prompt_mask, torch.zeros(len(item["box"]))
    
    def load_image(self, filename):
        return cv2.imread(filename)
    
    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def collate_fn(batch):
        samples, box, cls, prompt_mask, indices = zip(*batch)
        
        box = torch.cat(box, dim=0)
        cls = torch.cat(cls, dim=0)
        
        new_indices = list(indices)
        for i in range(len(indices)):
            new_indices[i] += i
        indices = torch.cat(new_indices, dim=0)

        targets = {'box': box,
                   'idx': indices,
                   'cls': cls}
        return torch.stack(samples, dim=0), torch.stack(prompt_mask, dim=0), targets
    
class LoadVisualPrompt:
    def __init__(self, nc=1):
        self.scale_factor = 1/8
        self.nc = nc
    
    def make_mask(self, boxes, h, w):
        x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
        r = torch.arange(w)[None, None, :]  # rows shape(1,1,w)
        c = torch.arange(h)[None, :, None]  # cols shape(1,h,1)

        return ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))
    
    def __call__(self, image, target_box, target_cls, target_mask=None):
        imgsz = image.shape[1:]
        masksz = (int(imgsz[0] * self.scale_factor), int(imgsz[1] * self.scale_factor))

        if target_box is not None:
            box = util.xywh2xyxy(target_box) * torch.tensor(masksz)[[1, 0, 1, 0]]  # target boxes
            masks = self.make_mask(box, *masksz).float()
        elif target_mask is not None:
            masks = torch.nn.functional.interpolate(target_mask.unsqueeze(1), masksz, mode="nearest").squeeze(1).float()
        else:
            raise ValueError("LoadVisualPrompt must have target_box or target_cls")
        cls = target_cls.squeeze(-1).to(torch.int)
        
        visuals = torch.zeros(self.nc, *masksz)
        cls_mask_dict = {}
        for idx, mask in zip(cls, masks):
            cls_mask_dict.setdefault(idx, []).append(mask)
        
        for idx, cls_masks in cls_mask_dict.items():
            sorted(cls_masks, key=lambda item: item.sum())
            for mask in cls_masks[:3]:
                visuals[idx] = torch.logical_or(visuals[idx], mask)

        return visuals
  