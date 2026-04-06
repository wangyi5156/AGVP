import logging
import os
from pathlib import Path
from av_nav.common.logging_utils import get_run_log_dir

_FUSION_LOGGER_NAME = "dynamic_fusion"
_FUSION_HANDLER_ATTACHED = False   # 防止多次重复加 handler

def init_fusion_logger(run_dir: str = None,
                       filename: str = "dynamic_fusion.log",
                       level: int = logging.INFO):
    """
    run_dir: 主训练日志所在目录；若为 None 则自动推断
    """
    global _FUSION_HANDLER_ATTACHED
    if _FUSION_HANDLER_ATTACHED:
        return

    log_dir = get_run_log_dir(run_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / filename

    logger = logging.getLogger(_FUSION_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False  # 避免跑到控制台

    # 清理旧 file handler（只清除我们同名的）
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)

    fh = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    fmt = logging.Formatter(
        fmt="%(asctime)s | FUSION | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    fh.setLevel(level)
    logger.addHandler(fh)

    logger.info("INIT fusion logger pid=%s run_dir=%s file=%s",
                os.getpid(), log_dir, log_path.name)
    _FUSION_HANDLER_ATTACHED = True

    return log_path 