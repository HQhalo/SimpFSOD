import numpy as np
import cv2
from dataset.resize import LetterBox
from utils import util
from dataset.dataset import LoadVisualPrompt
import cv2
from nets import nn
import subprocess
import torch 
import glob

letter_box = LetterBox(640)
vp_loader = LoadVisualPrompt()

def get_vpe(model, prompt_images):
    prompt_imgs = []
    prompt_masks = []
    for prompt_image in prompt_images:
        prompt_img = cv2.imread(prompt_image, cv2.IMREAD_UNCHANGED)
        mask_transparent = prompt_img[:, :, 3] == 0
        prompt_img[mask_transparent, :3] = 114
        prompt_img, support_box, prompt_cls, alpha = letter_box(prompt_img , np.array([[0,1,1,10,10]]), augment=False, coco_fotmat=True)
        prompt_img= prompt_img / 255.
        alpha = (alpha > 0).float().unsqueeze(0)
        prompt_mask = vp_loader(prompt_img, None, prompt_cls, alpha)
    
        prompt_mask = prompt_mask.cuda().half()
        prompt_img = prompt_img.cuda().half()
        prompt_masks.append(prompt_mask)
        prompt_imgs.append(prompt_img)

    support_vpe = model.get_vpe(torch.stack(prompt_imgs), torch.stack(prompt_masks))
    return support_vpe

def box_iou(box1, box2):
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

def merge_box_union(box1, box2):
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])
    return np.array([x1, y1, x2, y2, (box1[4] + box2[4])/2, box1[5]])

def merge_outputs_result(outputs, org_w = 1024, input_size = 640):  
    left_boxes = outputs[0].copy()
    right_boxes = outputs[1].copy()

    left_margin = org_w - input_size
    right_boxes[:, [0, 2]] += left_margin

    iou_mat = box_iou(left_boxes[:, :4], right_boxes[:, :4])

    merge_pairs = np.argwhere(iou_mat > 0.3)

    merged_left = set()
    merged_right = set()

    results = []

    for i, j in merge_pairs:
        results.append(merge_box_union(left_boxes[i], right_boxes[j]))
        merged_left.add(i)
        merged_right.add(j)

    for i in range(len(left_boxes)):
        if i not in merged_left:
            results.append(left_boxes[i])

    for j in range(len(right_boxes)):
        if j not in merged_right:
            results.append(right_boxes[j])

    return np.array(results)

def predict_frame(model, frames, vpe):
    input = []
    for frame in frames:
        query, _ , _= letter_box(frame, np.array([[-1, 1, 1,   1.,   1]]))
        input.append(query)
    query = torch.stack(input, dim=0)
    query = query / 255.
    query = query.cuda().half()
    re = model(query, vpe)
    outputs = util.non_max_suppression(re, conf_threshold=0.01, iou_threshold=0.4)
    result = []
    for output in outputs:
        output = output.detach().cpu().numpy()
        output[:,:4] = letter_box.unletter_box(output[:,:4])
        result.append(output)
    return result


def predict_video(model, video_path, output_path, vpe):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    batch_frame = []

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # batch_frame.append(frame)
        # if len(batch_frame) >= 32:
        #     outputs = predict_frame(model, batch_frame, vpe)
        #     # sorted(outputs, key=lambda item: item[4])
        #     for i in range(len(batch_frame)):
        #         tmp_frame = batch_frame[i]
        #         for box in outputs[i]:
        #             [x1,y1,x2,y2,conf, cls] = box
        #             cv2.rectangle(tmp_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        #             cv2.putText(tmp_frame, f'{conf:.4f}', (int(x1), max(0, int(y1) - 10)),
        #                         cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
        #         out.write(tmp_frame)
        #     batch_frame = []
        w = frame.shape[1]
        outputs = predict_frame(model, [frame[:,:640,:] , frame[:,-640:,:]], vpe)
        outputs = merge_outputs_result(outputs, org_w= w, input_size= 640)
        sorted(outputs, key=lambda item: item[4])
        for box in outputs[-3:]:
            [x1,y1,x2,y2,conf, cls] = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, f'{conf:.5f}', (int(x1), max(0, int(y1) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
        out.write(frame)
    
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(frame_idx)
        if frame_idx > 2000:
            break

    if len(batch_frame) > 0:
        outputs = predict_frame(model, batch_frame, vpe)
        # sorted(outputs, key=lambda item: item[4])
        for i in range(len(batch_frame)):
            tmp_frame = batch_frame[i]
            for box in outputs[i]:
                [x1,y1,x2,y2,conf, cls] = box
                cv2.rectangle(tmp_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(tmp_frame, f'{conf:.5f}', (int(x1), max(0, int(y1) - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            out.write(tmp_frame)

    cap.release()
    out.release()

model = nn.load_model("/home/quang/CODE/SimpFSOD/yoloe-v8s-pretrained.pt")
model.cuda()
model.half()
model.eval()

prompt_paths = glob.glob("/home/quang/DATA/public_test/samples/LifeJacket_0/object_images/*.png")
vpe = get_vpe(model, prompt_paths)
vpe = vpe.squeeze(1).unsqueeze(0)
# vpe = torch.mean(vpe, dim=0, keepdim=True)

predict_video(model, "/home/quang/DATA/public_test/samples/LifeJacket_1/drone_video.mp4", "/home/quang/CODE/SimpFSOD/video_test.mp4", vpe)

input_path = "/home/quang/CODE/SimpFSOD/video_test.mp4"
output_path = "/home/quang/CODE/SimpFSOD/video_test_fixed.mp4"

cmd = [
    "ffmpeg", "-y",
    "-i", input_path,
    "-vcodec", "libx264",
    "-crf", "18",
    "-preset", "veryfast",
    # "-vf", "scale=iw/2:ih/2",
    "-pix_fmt", "yuv420p",
    output_path
]

subprocess.run(cmd, check=True)