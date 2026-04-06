import argparse
import logging
import warnings
import torch
import os
import shutil

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from av_nav.common.baseline_registry import baseline_registry
from av_nav.config.default import get_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        default='train',
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        default='av_nav/config/pointgoal_rgb.yaml',
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Directory to save models and logs",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1,
        help="Evaluation interval of checkpoints",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action='store_true',
        help="Overwrite existing output directories",
    )
    parser.add_argument(
        "--prev-ckpt-ind",
        type=int,
        default=-1,
        help="Checkpoint index for evaluation",
    )
    args = parser.parse_args()

    config = get_config(args.exp_config, args.opts, args.model_dir, args.run_type, args.overwrite)

    # 打印当前使用的策略名，确保加载正确
    print(f"Using policy: {config.POLICY_NAME}")

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)

    torch.set_num_threads(1)

    level = logging.DEBUG if config.DEBUG else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    from habitat import logger
    logger.setLevel(logging.INFO)

    if args.run_type == "train":
        trainer.train()
    elif args.run_type == "eval":
        trainer.eval(args.eval_interval, args.prev_ckpt_ind, config.USE_LAST_CKPT)

if __name__ == "__main__":
    main()
