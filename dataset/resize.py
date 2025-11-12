import cv2
import numpy
import random
import torch
from dataset.transform import augment_hsv, random_perspective

def resample():
    choices = (cv2.INTER_AREA,
               cv2.INTER_CUBIC,
               cv2.INTER_LINEAR,
               cv2.INTER_NEAREST,
               cv2.INTER_LANCZOS4)
    return random.choice(seq=choices)

class LetterBox():
    def __init__(self, input_size, params={}, transform=None):
        self.input_size = input_size
        self.params = params
        self.transform = transform
        self.ratio = None
        self.pad = None

    def __call__(self, image, org_label, augment = False, coco_fotmat = False):
        """
        Resizes and pads an image for object detection
        Args:
            org_image (np.ndarray):     The input image as a numpy array. BGR format, size: [h, w, 3]
            org_label (np.ndarray):     The input box, YOLO format by default, size: [class_id, center_x, center_y, width, height], 
                                            set coco_fotmat = True to use COCO format

        Returns: 
            image (np.ndarray): RGB format, size: [3, h, w]
            box (np.ndarray):   YOLO format
        """
        h, w = image.shape[:2]
        label = org_label.copy()
        if coco_fotmat:
            label[:,1:] = coco2wh(label[:, 1:], w, h)

        box = label[:,1:] 
        cls = label[:,:1] 
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

        # Resize
        image, ratio, pad = self.resize(image, self.input_size, augment)
        self.ratio = ratio
        self.pad = pad
        alpha = None
        if image.shape[2] == 4:
            alpha = image[:, :, 3]
            image = image[:, :, :3]
        box = wh2xy(box, ratio[0] * w, ratio[1] * h, pad[0], pad[1])
        if augment:
            new_image, new_label = random_perspective(image.copy(), label.copy(), self.params)
            if len(new_label) > 0:
                image, label = new_image, new_label        
        box = xy2wh(box, self.input_size, self.input_size)

        # Convert HWC to CHW, BGR to RGB
        sample = image.transpose((2, 0, 1))[::-1]
        sample = numpy.ascontiguousarray(sample)
        if alpha is None:
            return torch.from_numpy(sample), torch.from_numpy(box), torch.from_numpy(cls)
        else:
            return torch.from_numpy(sample), torch.from_numpy(box), torch.from_numpy(cls), torch.from_numpy(alpha)

    
    def albumentations(self, image, box):
        x = self.transform(image=image,
                            bboxes=box)
        image = x['image']
        box = numpy.array(x['bboxes'])
        return image, box
    
    def resize(self, image, input_size, augment):
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
        border_value = (114, 114, 114, 0) if image.shape[2] == 4 else (114, 114, 114)
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_value)  # add border
        return image, (r, r), (w, h)

    def convert_box(self, x):
        return xyxy2xyxy(x, self.ratio[0], self.ratio[1], self.pad[0], self.pad[1])
    
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

def xyxy2xyxy(x, r_w, r_h, pad_w, pad_h):
    y = numpy.copy(x)
    y[:0] = (x[:, 0] - pad_w) / r_w
    y[:1] = (x[:, 0] - pad_h) / r_h
    y[:2] = (x[:, 2] - pad_w) / r_w
    y[:3] = (x[:, 3] - pad_h) / r_h

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
