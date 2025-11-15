import copy
import csv
import os
import warnings

import torch
import tqdm
from torch.utils import data

from nets import nn, loss
from utils import util
from dataset.dataset import VPDataset, ZaloVPDataset, SyntheticVPDataset

warnings.filterwarnings("ignore")



def train(args, params):
    util.setup_seed()
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using device: {device}")

    # Model
    model = nn.load_model("/home/quang/CODE/SimpFSOD/yoloe-v8s-pretrained.pt", 1)

    for name, param in model.net.named_parameters():
        param.requires_grad = False
    for name, param in model.fpn.named_parameters():
        param.requires_grad = False
    for name, param in model.head.box.named_parameters():
        param.requires_grad = False
    for name, param in model.head.emb.named_parameters():
        param.requires_grad = False
    for name, param in model.head.dfl.named_parameters():
        param.requires_grad = False
    for name, param in model.head.bn.named_parameters():
        param.requires_grad = False
    # for name, param in model.head.savpe.named_parameters():
    #     param.requires_grad = False

    print(params)

    model.cuda()

    # Optimizer
    accumulate = max(round(32 / args.batch_size), 1)
    params['weight_decay'] *= args.batch_size * accumulate / 32

    p = [], [], []
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, torch.nn.Parameter):
            p[2].append(v.bias)
        if isinstance(v, torch.nn.BatchNorm2d):
            p[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, torch.nn.Parameter):
            p[0].append(v.weight)

    optimizer = torch.optim.AdamW([
            {'params': p[0], 'weight_decay': params['weight_decay']},
            {'params': p[1], 'weight_decay': 0.0},
            {'params': p[2], 'weight_decay': 0.0},
        ], lr= params['min_lr'])
    del p

    # EMA
    ema = util.EMA(model)

    # Dataset
    # dataset = VPDataset("/home/quang/DATA/got10k/train", 640, params, augment=True)
    # loader = data.DataLoader(dataset, args.batch_size, True, num_workers=8, pin_memory=True)
    
    # dataset_test = VPDataset("/home/quang/DATA/got10k/val", 640, params, augment=False)
    # loader_test = data.DataLoader(dataset_test, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    folder = "/home/quang/DATA/zalo_dataset"
    videos = [entry.name for entry in os.scandir(folder) if entry.is_dir()]
    dataset_zalo = ZaloVPDataset(folder, 640, params, augment=False, videos=videos, sample_down=5)
    loader = data.DataLoader(dataset_zalo, args.batch_size, True, num_workers=8, pin_memory=True,
                             collate_fn=ZaloVPDataset.collate_fn)
    
    # dataset_syn = SyntheticVPDataset("/home/quang/DATA/synthtic_dataset", 640, params, augment=False)
    
    # dataset = data.ConcatDataset([dataset_syn, dataset_zalo])

    # loader = data.DataLoader(dataset_syn, args.batch_size, True, num_workers=8, pin_memory=True,
    #                          collate_fn=SyntheticVPDataset.collate_fn)

    dataset_test = ZaloVPDataset(folder, 640, params, augment=False, videos=videos[-2:])
    loader_test = data.DataLoader(dataset_test, batch_size=4, shuffle=False, num_workers=4,
                             pin_memory=True, collate_fn=ZaloVPDataset.collate_fn)

    # Scheduler
    num_steps = len(loader)
    scheduler = util.CosineLR(args, params, num_steps)

    # Start training
    best = 0
    amp_scale = torch.amp.GradScaler()
    criterion = loss.ComputeLoss(model, params)

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
        for i, (sample, prompt_img, prompt_mask, targets) in p_bar:
            step = i + num_steps * epoch
            scheduler.step(step, optimizer)

            sample = sample.cuda().float() / 255
            prompt_img = prompt_img.cuda().float() / 255
            prompt_mask = prompt_mask.cuda()

            # Forward
            with torch.amp.autocast(device_type=device):
                vpe = model.get_vpe(prompt_img, prompt_mask)
                outputs = model(sample, vpe)  # forward
            loss_box, loss_cls, loss_dfl = criterion(outputs, targets)

            avg_box_loss.update(loss_box.item(), sample.size(0))
            avg_cls_loss.update(loss_cls.item(), sample.size(0))
            avg_dfl_loss.update(loss_dfl.item(), sample.size(0))

            loss_box *= args.batch_size  # loss scaled by batch_size
            loss_cls *= args.batch_size  # loss scaled by batch_size
            loss_dfl *= args.batch_size  # loss scaled by batch_size

            # Backward
            amp_scale.scale(loss_box + loss_cls+ loss_dfl).backward()

            if step % accumulate == 0:
                # amp_scale.unscale_(optimizer)  # unscale gradients
                # util.clip_gradients(model)  # clip gradients
                amp_scale.step(optimizer)  # optimizer.step
                amp_scale.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
            
            torch.cuda.synchronize()

            # Log
            memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G'  # (GB)
            s = ('%10s' * 2 + '%10.5g' * 3) % (f'{epoch + 1}/{args.epochs}', memory,
                                                avg_box_loss.avg, avg_cls_loss.avg, avg_dfl_loss.avg)
            p_bar.set_description(s)


        # mAP
        last = test(args, params, ema.ema, loader_test)

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
def test(args, params, model=None, loader=None):
    if loader == None:
        folder = "/home/quang/DATA/zalo_dataset"
        videos = [entry.name for entry in os.scandir(folder) if entry.is_dir()]
        dataset_test = ZaloVPDataset(folder, 640, params, augment=False, videos=videos[-2:])
        loader = data.DataLoader(dataset_test, batch_size=16, shuffle=False, num_workers=4,
                                pin_memory=True, collate_fn=ZaloVPDataset.collate_fn)

    if not model:
        # model = torch.load(f='./weights/best.pt', map_location='cuda')
        # model = model['model'].float().fuse()
        model = nn.load_model("/home/quang/CODE/SimpFSOD/zalo-best.pt")
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
    for samples, prompt_img, prompt_mask, targets in p_bar:
        samples = samples.cuda()
        # samples = samples.half()  # uint8 to fp16/32
        samples = samples / 255.  # 0 - 255 to 0.0 - 1.0
        _, _, h, w = samples.shape  # batch-size, channels, height, width
        scale = torch.tensor((w, h, w, h)).cuda()
        
        prompt_img = prompt_img.cuda()
        prompt_img = prompt_img / 255.
        # prompt = prompt.half()

        prompt_mask = prompt_mask.cuda()
        # prompt_mask = prompt_mask.half()

        # Inference
        vpe = model.get_vpe(prompt_img, prompt_mask)
        outputs = model(samples, vpe) 
        # NMS
        # outputs = util.non_max_suppression(outputs, conf_threshold=0.05, iou_threshold=0.6)
        outputs = util.non_max_suppression(outputs, conf_threshold=0.2, iou_threshold=0.4)
        # Metrics
        for i, output in enumerate(outputs):
            idx = targets['idx'] == i
            cls = targets['cls'][idx]
            box = targets['box'][idx]

            cls = cls.cuda()
            box = box.cuda()

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
    print(('%10s' + '%10.5g' * 4) % ('', m_pre, m_rec, map50, mean_ap))
    # Return results
    model.float()  # for training
    return mean_ap, map50, m_rec, m_pre
