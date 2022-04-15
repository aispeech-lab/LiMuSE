import torch
import numpy as np
import random
import argparse
import sys
sys.path.append('./options')
from trainer import Trainer
from LiMuSE import LiMuSE
from options.option import parse
import utils
from prepareMultiCueDataOnGrid import PrepareMultiCueGridDataSamples

seed = 2021

def main():
    # Reading option
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default='/mnt/lustre/xushuang4/liuqinghua/LiMuSE/options/train/train.yml', help='Path to option YAML file.')
    args = parser.parse_args()

    opt = parse(args.opt, is_tain=True)
    logger = utils.get_logger(__name__)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    logger.info('Building the model of LiMuSE')
    net = LiMuSE(**opt['net_conf'])

    logger.info('Building the trainer of LiMuSE')
    gpuid = tuple(opt['gpu_ids'])
    trainer = Trainer(net, **opt['train'], resume=opt['resume'],
                      gpuid=gpuid, optimizer_kwargs=opt['optimizer_kwargs'])

    logger.info('Making the train and test data loader')
    config = utils.read_config('/mnt/lustre/xushuang4/liuqinghua/LiMuSE/options/train/train.yml')
    grid_samples = PrepareMultiCueGridDataSamples(config)
  
    trainer.run(grid_samples)


if __name__ == "__main__":
    main()
