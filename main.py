import argparse
import os
import warnings

import torch
import yaml
import thop

from nets import nn, loss
from utils import util
from train import trainer, small_obj_trainer

warnings.filterwarnings("ignore")

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
    parser.add_argument('--epochs', default=4, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--small-obj', action='store_true')
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
        trainer.train(args, params)
    if args.small_obj:
        small_obj_trainer.train(args, params)
    if args.test:
        trainer.test(args, params)

if __name__ == "__main__":
    main()