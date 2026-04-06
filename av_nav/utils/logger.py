# av_nav/utils/logger.py
import logging
import os


def build_logger(name: str,
                 log_file: str,
                 level: str = "INFO",
                 fmt: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                 datefmt: str = "%Y-%m-%d %H:%M:%S",
                 overwrite: bool = False):
    """
    统一构造 logger:
      - 避免重复添加 handler
      - 正确使用 datefmt 而不是在 fmt 中写 %Y-%m-%d
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False  # 避免冒泡到 root 造成重复输出

    # 如果要覆盖旧文件
    if overwrite and os.path.isfile(log_file):
        try:
            os.remove(log_file)
        except OSError:
            pass

    # 检查是否已经有同类型的 FileHandler，避免重复
    need_file = True
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "_target_file", None) == os.path.abspath(log_file):
            need_file = False
            break

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    if need_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(formatter)
        # 记录这个 handler 的目标文件以防重复
        fh._target_file = os.path.abspath(log_file)
        logger.addHandler(fh)

    # 也可以添加一个控制台 handler（如果需要）
    need_console = True
    for h in logger.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            need_console = False
            break
    if need_console:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger