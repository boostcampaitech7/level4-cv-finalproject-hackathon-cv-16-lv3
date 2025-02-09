import os
import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb

from utils import *
from config import Config
from dist_utils import get_rank, init_distributed_mode
from models import load_model
from dataset import SALMONNDataset
from runner import Runner

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument("--cfg-path", type=str, required=True, help='path to configuration file')
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--dryrun", action='store_true', help='if True, use dummy model and skip forward/backward')
    parser.add_argument("--log-level", type=str, default="INFO", help='logging level')  # 디버깅을 위해 추가.

    return parser.parse_args()


def setup_seeds(config):
    seed = config.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def main():
    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    # load config
    args = parse_args()
    cfg = Config(args)
    run_config = cfg.config.run
    model_config = cfg.config.model
    data_config = cfg.config.datasets

    # initialize distributed training
    init_distributed_mode(run_config)
    setup_seeds(run_config)
    setup_logger() # set after init_distributed_mode() to only log on master.

    
    
    # Wandb logger
    global_rank = int(os.environ["RANK"])
    if global_rank == 0:
        wandb.login()
        wandb.init(entity="CV_SOTA", project="audio_lm", name=run_config.exp_name)



    # print config
    cfg.pretty_print()

    # build datasets
    if "augmentation" not in data_config:
        datasets = {
            "train": SALMONNDataset(data_config.prefix, data_config.train_ann_path, data_config.whisper_path),
            "valid": SALMONNDataset(data_config.prefix, data_config.valid_ann_path, data_config.whisper_path),
            "test": SALMONNDataset(data_config.prefix, data_config.test_ann_path, data_config.whisper_path),
        }
    else:
        datasets = {
            "train": SALMONNDataset(data_config.prefix, data_config.train_ann_path, data_config.whisper_path, data_config.augmentation),
            "valid": SALMONNDataset(data_config.prefix, data_config.valid_ann_path, data_config.whisper_path),
            "test": SALMONNDataset(data_config.prefix, data_config.test_ann_path, data_config.whisper_path),
        }

    # dataset build 완료
    print("datasets build 를 완료하였습니다.")
    
    # 모델 빌드
    if not args.dryrun:
        model = load_model(model_config)
        print("실제 모델이 로드되었습니다.")
    else:  # 작은 더미 언어 모델 로드
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("apple/OpenELM-270M-Instruct", trust_remote_code=True)
        print("더미 모델이 로드되었습니다.")

    # 러너 빌드
    runner = Runner(cfg, model, datasets, job_id, args.dryrun)
    print("러너가 성공적으로 빌드되었습니다.")

    # train
    runner.train()


if __name__ == "__main__":
    main()