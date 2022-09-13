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

seed = 2018

def main():
    # Reading option
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default='/path/to/train.yml', help='Path to option YAML file.')
    parser.add_argument('-train', type=bool, default=True, help='train or test.') # True for train, False for test
    args = parser.parse_args()

    opt = parse(args.opt, is_tain=True)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print('Building the model of LiMuSE')
    net = LiMuSE(**opt['net_conf'])

    print('Building the trainer of LiMuSE')
    gpuid = tuple(opt['gpu_ids'])
    trainer = Trainer(net, **opt['train'], resume=opt['resume'],
                      gpuid=gpuid, optimizer_kwargs=opt['optimizer_kwargs'])

    print('Making the train and test data loader')
    config = utils.read_config(args.opt)
    print('Config', config)
    grid_samples = PrepareMultiCueGridDataSamples(config)
  
    if args.train:
        trainer.run(grid_samples)
        trainer.test(grid_samples)
    else:
        trainer.test(grid_samples)


if __name__ == "__main__":
    main()
