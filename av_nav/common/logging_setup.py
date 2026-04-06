import logging
import os

def init_dynamic_fusion_logger(log_dir="logs", filename="dynamic_fusion.log", level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.abspath(os.path.join(log_dir, filename))

    logger = logging.getLogger("dynamic_fusion")
    logger.setLevel(level)

    # 清理旧 FileHandler（避免重复添加）
    to_remove = []
    for h in logger.handlers:
        from logging import FileHandler
        if isinstance(h, FileHandler):
            to_remove.append(h)
    for h in to_remove:
        logger.removeHandler(h)

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fmt = logging.Formatter(
        "%(asctime)s | FUSION | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(fmt)
    fh.setLevel(level)
    logger.addHandler(fh)

    # 不要向上冒泡到 root（避免进控制台）
    logger.propagate = False

    # 立即写一条探针
    logger.info("=== dynamic_fusion logger initialized at %s ===", log_path)

    return logger, log_path 