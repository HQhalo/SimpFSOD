import argparse
import copy
import csv
import os
import warnings

import torch
import tqdm
import yaml
from torch.utils import data
import thop

from nets import nn, loss
from utils import util
from dataset.dataset import VPDataset

warnings.filterwarnings("ignore")



def train(args, params):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using device: {device}")

    # Model
    model = nn.load_model("/home/quang/CODE/SimpFSOD/v8_s.pt")

    for name, param in model.net.named_parameters():
        param.requires_grad = False
    for name, param in model.fpn.named_parameters():
        param.requires_grad = False

    model.cuda()

    # Optimizer
    p = [], [], []
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, torch.nn.Parameter):
            p[2].append(v.bias)
        if isinstance(v, torch.nn.BatchNorm2d):
            p[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, torch.nn.Parameter):
            p[0].append(v.weight)

    optimizer = torch.optim.SGD(p[2], params['min_lr'], params['momentum'], nesterov=True)
    optimizer.add_param_group({'params': p[0], 'weight_decay': params['weight_decay']})
    optimizer.add_param_group({'params': p[1]})
    del p

    # EMA
    ema = util.EMA(model)

    # Dataset
    dataset = VPDataset("/home/quang/DATA/got10k/train", 640, params, augment=True)
    loader = data.DataLoader(dataset, args.batch_size, True, num_workers=8, pin_memory=True)

    # Scheduler
    num_steps = len(loader)
    scheduler = util.LinearLR(args, params, num_steps)

    # Start training
    best = 0
    amp_scale = torch.amp.GradScaler()
    criterion = loss.ComputeLoss(model, params)
    with open('weights/step.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch',
                                                    'box', 'cls', 'dfl',
                                                    'Recall', 'Precision', 'mAP@50', 'mAP'])
        writer.writeheader()
        for epoch in range(args.epochs):
            model.train()

            if args.epochs - epoch == 10:
                loader.dataset.mosaic = False

            avg_box_loss = loss.AverageMeter()
            avg_cls_loss = loss.AverageMeter()
            avg_dfl_loss = loss.AverageMeter()
            optimizer.zero_grad()

            p_bar = enumerate(loader)
            print(('\n' + '%10s' * 5) % ('epoch', 'memory', 'box', 'cls', 'dfl'))
            p_bar = tqdm.tqdm(p_bar, total=num_steps, dynamic_ncols=False, ncols=100)
            for i, (sample, box, prompt, prompt_mask) in p_bar:
                step = i + num_steps * epoch
                scheduler.step(step, optimizer)

                sample = sample.cuda().float() / 255
                prompt = prompt.cuda().float() / 255
                prompt_mask = prompt_mask.cuda()

                # Forward
                with torch.amp.autocast(device_type=device):
                    vpe = model.get_vpe(prompt, prompt_mask)
                    outputs = model(sample, vpe)  # forward
                loss_box, loss_cls, loss_dfl = criterion(outputs, box)

                avg_box_loss.update(loss_box.item(), sample.size(0))
                avg_cls_loss.update(loss_cls.item(), sample.size(0))
                avg_dfl_loss.update(loss_dfl.item(), sample.size(0))

                loss_box *= args.batch_size  # loss scaled by batch_size
                loss_cls *= args.batch_size  # loss scaled by batch_size
                loss_dfl *= args.batch_size  # loss scaled by batch_size

                # Backward
                amp_scale.scale(loss_box + loss_cls + loss_dfl).backward()

                # Optimize
                # amp_scale.unscale_(optimizer)  # unscale gradients
                # util.clip_gradients(model)  # clip gradients
                amp_scale.step(optimizer)  # optimizer.step
                amp_scale.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

                # Log
                memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G'  # (GB)
                s = ('%10s' * 2 + '%10.3g' * 3) % (f'{epoch + 1}/{args.epochs}', memory,
                                                    avg_box_loss.avg, avg_cls_loss.avg, avg_dfl_loss.avg)
                p_bar.set_description(s)


            # mAP
            last = test(args, params, ema.ema)

            writer.writerow({'epoch': str(epoch + 1).zfill(3),
                                'box': str(f'{avg_box_loss.avg:.3f}'),
                                'cls': str(f'{avg_cls_loss.avg:.3f}'),
                                'dfl': str(f'{avg_dfl_loss.avg:.3f}'),
                                'mAP': str(f'{last[0]:.3f}'),
                                'mAP@50': str(f'{last[1]:.3f}'),
                                'Recall': str(f'{last[2]:.3f}'),
                                'Precision': str(f'{last[3]:.3f}')})
            f.flush()

            # Update best mAP
            if last[0] > best:
                best = last[0]

            # Save model
            save = {'epoch': epoch + 1,
                    'model': copy.deepcopy(ema.ema)}

            # Save last, best and delete
            torch.save(save, f='./weights/last.pt')
            if best == last[0]:
                torch.save(save, f='./weights/best.pt')
            del save

    util.strip_optimizer('./weights/best.pt')  # strip optimizers
    util.strip_optimizer('./weights/last.pt')  # strip optimizers
    torch.cuda.empty_cache()


@torch.no_grad()
def test(args, params, model=None):
    dataset = VPDataset("/home/quang/DATA/got10k/val", 640, params, augment=False)
    # dataset = ZaloVPDataset("/home/quang/DATA/zalo_dataset", 640, params, augment=False)

    loader = data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4,
                             pin_memory=True)

    if not model:
        # model = torch.load(f='./weights/best.pt', map_location='cuda')
        # model = model['model'].float().fuse()
        model = nn.load_model("/home/quang/CODE/SimpFSOD/v8_s.pt")
        model = model.cuda()


    # model.half()
    model.eval()

    # Configure
    iou_v = torch.linspace(start=0.5, end=0.95, steps=10).cuda()  # iou vector for mAP@0.5:0.95
    n_iou = iou_v.numel()

    m_pre = 0
    m_rec = 0
    map50 = 0
    mean_ap = 0
    metrics = []
    p_bar = tqdm.tqdm(loader, desc=('%10s' * 5) % ('', 'precision', 'recall', 'mAP50', 'mAP'), dynamic_ncols=False, ncols=100)
    for samples, box_target, prompt, prompt_mask in p_bar:
        samples = samples.cuda()
        # samples = samples.half()  # uint8 to fp16/32
        samples = samples / 255.  # 0 - 255 to 0.0 - 1.0
        _, _, h, w = samples.shape  # batch-size, channels, height, width
        scale = torch.tensor((w, h, w, h)).cuda()
        
        prompt = prompt.cuda()
        prompt = prompt / 255.
        # prompt = prompt.half()

        prompt_mask = prompt_mask.cuda()
        # prompt_mask = prompt_mask.half()

        # Inference
        vpe = model.get_vpe(prompt, prompt_mask)
        outputs = model(samples, vpe) 
        # NMS
        outputs = util.non_max_suppression(outputs)
        # Metrics
        for i, output in enumerate(outputs):
            cls = torch.zeros(1,1).cuda()
            box = box_target[i].cuda()

            metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()
            if output.shape[0] == 0:
                if cls.shape[0]:
                    metrics.append((metric, *torch.zeros((2, 0)).cuda(), cls.squeeze(-1)))
                continue
            # Evaluate
            if cls.shape[0]:
                target = torch.cat(tensors=(cls, util.wh2xy(box) * scale), dim=1)
                metric = util.compute_metric(output[:, :6], target, iou_v)
            # Append
            metrics.append((metric, output[:, 4], output[:, 5], cls.squeeze(-1)))

    # Compute metrics
    metrics = [torch.cat(x, dim=0).cpu().numpy() for x in zip(*metrics)]  # to numpy
    if len(metrics) and metrics[0].any():
        tp, fp, m_pre, m_rec, map50, mean_ap = util.compute_ap(*metrics)
    # Print results
    print(('%10s' + '%10.3g' * 4) % ('', m_pre, m_rec, map50, mean_ap))
    # Return results
    model.float()  # for training
    return mean_ap, map50, m_rec, m_pre

def profile(args, params):
    shape = (1, 3, args.input_size, args.input_size)
    model = nn.yolo_v8_s().fuse()

    model.eval()
    model(torch.zeros(shape), torch.zeros(1, 1, 512))

    x = torch.empty(shape)
    y = torch.empty(1, 1, 512)
    flops, num_params = thop.profile(model, inputs=[x, y], verbose=False)
    flops, num_params = thop.clever_format(nums=[2 * flops, num_params], format="%.3f")

    print(f'Number of parameters: {num_params}')
    print(f'Number of FLOPs: {flops}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    if not os.path.exists('weights'):
        os.makedirs('weights')

    util.setup_seed()
    util.setup_multi_processes()

    with open(os.path.join('utils', 'args.yaml'), errors='ignore') as f:
        params = yaml.safe_load(f)
    
    profile(args, params)
    if args.train:
        train(args, params)
    if args.test:
        test(args, params)


if __name__ == "__main__":
    main()