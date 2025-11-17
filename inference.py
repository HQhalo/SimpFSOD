import numpy as np
from dataset.resize import LetterBox
from utils import util
from dataset.dataset import LoadVisualPrompt
import cv2
from nets import nn
import json
import cv2
import torch 
import glob
from tqdm import tqdm
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
import os

class VideoPredictor():
    def __init__(self, model, input_size=640):
        self.letter_box = LetterBox(input_size)
        self.vp_loader = LoadVisualPrompt()
        self.embedding = Embedding()
        self.model = model
        self.input_size = input_size
        self.conf_threshold = 0.01
        self.iou_threshold = 0.4


    def get_vpe(self, prompt_images):
        prompt_imgs = []
        prompt_masks = []
        for prompt_image in prompt_images:
            prompt_img = cv2.imread(prompt_image, cv2.IMREAD_UNCHANGED)
            mask_transparent = prompt_img[:, :, 3] == 0
            prompt_img[mask_transparent, :3] = 114
            prompt_img, support_box, prompt_cls, alpha = self.letter_box(prompt_img , np.array([[0,1,1,10,10]]), augment=False, coco_fotmat=True)
            prompt_img= prompt_img / 255.
            alpha = (alpha > 0).float().unsqueeze(0)
            prompt_mask = self.vp_loader(prompt_img, None, prompt_cls, alpha)
        
            prompt_mask = prompt_mask.cuda().half()
            prompt_img = prompt_img.cuda().half()
            prompt_masks.append(prompt_mask)
            prompt_imgs.append(prompt_img)

        support_vpe = self.model.get_vpe(torch.stack(prompt_imgs), torch.stack(prompt_masks))
        return support_vpe
    def predict_video(self, video_path, vpe, prompt_embs, output_path=None):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        result = {}
        for frame_idx in tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret:
                break
            # if frame_idx < 1020:
            #     continue

            w = frame.shape[1]
            outputs = self.predict_frame([frame[:,:640,:] , frame[:,-640:,:]], vpe)
            outputs = self.merge_outputs_result(outputs, org_w= w, input_size= 640)

            # sorted(outputs, key=lambda item: item[4])

            crops = []
            for box in outputs:
                [x1,y1,x2,y2,conf, cls] = box
                crop = frame[int(max(0, y1)):int(max(0,y2)), int(max(0,x1)):int(max(0,x2))]
                crops.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))

            if len(crops) > 0:
                crops_embs = self.embedding(crops)   
                sims = torch.matmul(crops_embs, prompt_embs.T)
                sim_array = sims.mean(dim =1).detach().cpu()

                sim_idx = sims.mean(dim =1).argmax().detach().cpu().item()
                conf_idx = outputs[:,4].argmax().item()
                box_result = []
                for i in range(len(crops)):
                # if sim_array[sim_idx].item() > 0.05:
                    [x1,y1,x2,y2,conf, cls] = outputs[i]
                    box_result.append({"frame": frame_idx, "conf": float(conf), "sim": float(sim_array[i]), "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)})
                    # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    # cv2.putText(frame, f'{conf:.5f}', (int(x1), max(0, int(y1) - 10)),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                    # cv2.putText(frame, f'{sim_array[sim_idx].item():.5f}', (int(x1), max(0, int(y1) + 30)),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                    
                result[frame_idx] = box_result
            # cv2.putText(frame, f'{frame_idx}', (0, 60),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                
            # out.write(frame)
            # if frame_idx > 4000:
            #     break

        cap.release()
        # out.release()
        return result
    def predict_frame(self, frames, vpe):
        input = []
        for frame in frames:
            query, _ , _= self.letter_box(frame, np.array([[-1, 1, 1,   1.,   1]]))
            input.append(query)
        query = torch.stack(input, dim=0)
        query = query / 255.
        query = query.cuda().half()
        re = self.model(query, vpe)
        outputs = util.non_max_suppression(re, conf_threshold=self.conf_threshold, iou_threshold=self.iou_threshold)
        result = []
        for output in outputs:
            output = output.detach().cpu().numpy()
            output[:,:4] = self.letter_box.unletter_box(output[:,:4])
            result.append(output)
        return result

    def box_iou(self, box1, box2):
        box1 = np.asarray(box1)
        box2 = np.asarray(box2)

        # expand dims for broadcast
        # box1[:, None] -> [N,1,4]
        # box2[None]    -> [1,M,4]
        a1 = box1[:, None, :2]  # top-left
        a2 = box1[:, None, 2:]  # bottom-right
        b1 = box2[None, :, :2]
        b2 = box2[None, :, 2:]

        # intersection
        inter_wh = np.minimum(a2, b2) - np.maximum(a1, b1)
        inter_wh = np.clip(inter_wh, 0, None)  # clamp(0)
        inter = inter_wh[..., 0] * inter_wh[..., 1]  # [N, M]

        # area box1, box2
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M]

        # IoU
        return inter / (area1[:, None] + area2 - inter + 1e-9)

    def merge_box_union(self, box1, box2):
        x1 = min(box1[0], box2[0])
        y1 = min(box1[1], box2[1])
        x2 = max(box1[2], box2[2])
        y2 = max(box1[3], box2[3])
        return np.array([x1, y1, x2, y2, (box1[4] + box2[4])/2, box1[5]])

    def merge_outputs_result(self, outputs, org_w = 1024, input_size = 640):  
        left_boxes = outputs[0].copy()
        right_boxes = outputs[1].copy()

        left_margin = org_w - input_size
        right_boxes[:, [0, 2]] += left_margin

        iou_mat = self.box_iou(left_boxes[:, :4], right_boxes[:, :4])

        merge_pairs = np.argwhere(iou_mat > 0.3)

        merged_left = set()
        merged_right = set()

        results = []

        for i, j in merge_pairs:
            results.append(self.merge_box_union(left_boxes[i], right_boxes[j]))
            merged_left.add(i)
            merged_right.add(j)

        for i in range(len(left_boxes)):
            if i not in merged_left:
                results.append(left_boxes[i])

        for j in range(len(right_boxes)):
            if j not in merged_right:
                results.append(right_boxes[j])

        return np.array(results)

class Embedding():
    def __init__(self):
        modelEfficient = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        modelEfficient.eval()

        feature_extractor = torch.nn.Sequential(*list(modelEfficient.children())[:-1])
        feature_extractor = feature_extractor.half()
        feature_extractor = feature_extractor.cuda()
        self.extractor = feature_extractor
        self.preprocesor = self.create_preprocess()

    def create_preprocess(self, size=80):
        return transforms.Compose([
            transforms.Resize((size,size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                std=[0.229,0.224,0.225])
        ])

    def __call__(self, images):
        images = [self.preprocesor(image.convert("RGB")) for image in images]
        images = torch.stack(images).half().cuda()
        with torch.no_grad():
            feat = self.extractor(images)
            return F.normalize(feat.flatten(start_dim=1), dim=1)
    # return F.cosine_similarity(feat1, feat2)



model = nn.load_model("/home/quang/CODE/SimpFSOD/yoloe-v8s-pretrained.pt")
model.cuda()
model.half()
model.eval()

video_predictor = VideoPredictor(model)
folder = "/home/quang/DATA/public_test/samples"
data = {}
for video_id in os.listdir(folder):
    print(video_id)
    prompt_resample_paths = glob.glob(f"{folder}/{video_id}/object_images/*_resample.png")
    prompt_bg_paths = glob.glob(f"{folder}/{video_id}/object_images/*_bg.png")
    print(f"Total prompt resample image: {len(prompt_resample_paths)}")

    vpe = video_predictor.get_vpe(prompt_resample_paths)
    vpe = vpe.squeeze(1).unsqueeze(0)
    prompt_embs = video_predictor.embedding([Image.open(prompt_path) for prompt_path in prompt_bg_paths ])

    result = video_predictor.predict_video(f"{folder}/{video_id}/drone_video.mp4", vpe, prompt_embs, "/home/quang/CODE/SimpFSOD/video_test.mp4")
    data[video_id] = result
with open('/home/quang/CODE/SimpFSOD/data.json', 'w') as f:
    json.dump(data, f)
# input_path = "/home/quang/CODE/SimpFSOD/video_test.mp4"
# output_path = "/home/quang/CODE/SimpFSOD/video_test_fixed.mp4"

# cmd = [
#     "ffmpeg", "-y",
#     "-i", input_path,
#     "-vcodec", "libx264",
#     "-crf", "18",
#     "-preset", "veryfast",
#     # "-vf", "scale=iw/2:ih/2",
#     "-pix_fmt", "yuv420p",
#     output_path
# ]

# subprocess.run(cmd, check=True)


