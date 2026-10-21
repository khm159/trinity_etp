import hydra
from trinity.tools.evaluator import Evaluator
import random
import torch
import numpy as np

@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg):

    # set random seed
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    evaluator = Evaluator(cfg)
    evaluator.evaluate()

if __name__=='__main__':
    main()