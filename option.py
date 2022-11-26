import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--multigpu", type=bool, default=True)
    parser.add_argument("--device", type=str, default="0")

    # models
    parser.add_argument("--pretrained", type=bool, default=False)
    parser.add_argument("--input_size", type=int, default=240) 

    # dataset
    parser.add_argument("--data_dir", type=str, default="./datasets/")
    parser.add_argument("--data_name", type=str, default='cars')

    # training setting
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--n_epoch", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--eval_epoch", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)

    # misc
    parser.add_argument("--ckpt_root", type=str, default="./FT_model")

    return parser.parse_args()


def make_template(opt):
    # device
    opt.device_ids = [int(item) for item in opt.device.split(',')]
    if len(opt.device_ids) == 1:
        opt.multigpu = False
    opt.gpu = opt.device_ids[0]


def get_option():
    opt = parse_args()
    make_template(opt)
    return opt