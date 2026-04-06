import logging
import os
from typing import Dict, Any


def get_dynfusion_logger(log_dir: str = "logs", filename: str = "dynamic_fusion.log"):
    """
    创建专门的dynamic fusion日志记录器，避免输出到控制台
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("dynfusion")
    
    # 防止重复添加handler
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(log_dir, filename), mode="a", encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)
        # 不向上冒泡到root，避免出现在控制台
        logger.propagate = False
    
    return logger


def init_dynfusion_logger(log_path="logs/dynamic_fusion.log"):
    """
    [DYNFUSION_LOG] 初始化独立的dynfusion logger
    """
    import os
    import logging
    
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    lg = logging.getLogger("dynfusion")
    
    # 防止重复添加
    if lg.handlers:
        return lg
    
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    lg.addHandler(fh)
    lg.setLevel(logging.INFO)
    lg.propagate = False
    lg.info("logger initialized OK")
    return lg


def format_fusion_stats(update_id: int, fusion_stats: Dict[str, Any], ppo_stats: Dict[str, Any]) -> str:
    """
    格式化fusion和PPO统计信息为日志字符串
    """
    # 从fusion_stats获取alpha相关统计
    alpha_mean = fusion_stats.get("alpha_mean", float("nan"))
    alpha_min = fusion_stats.get("alpha_min", float("nan"))
    alpha_max = fusion_stats.get("alpha_max", float("nan"))
    alpha_std = fusion_stats.get("alpha_std", float("nan"))
    gate_sparse = fusion_stats.get("gate_sparse", float("nan"))
    
    # 从ppo_stats获取PPO相关统计
    ratio_mean = ppo_stats.get("ratio_mean", float("nan"))
    clip_frac = ppo_stats.get("clip_frac", float("nan"))
    approx_kl = ppo_stats.get("approx_kl", float("nan"))
    dist_entropy = ppo_stats.get("dist_entropy", float("nan"))
    value_loss = ppo_stats.get("value_loss", float("nan"))
    action_loss = ppo_stats.get("action_loss", float("nan"))
    adv_mean = ppo_stats.get("adv_mean", float("nan"))
    adv_std = ppo_stats.get("adv_std", float("nan"))
    ret_mean = ppo_stats.get("ret_mean", float("nan"))
    value_pred_mean = ppo_stats.get("value_pred_mean", float("nan"))
    grad_norm_total = ppo_stats.get("grad_norm_total", float("nan"))
    feat_grad_mean = ppo_stats.get("feat_grad_mean", float("nan"))
    
    return (
        f"[UPD {update_id}] alpha(mean={alpha_mean:.4f} min={alpha_min:.4f} "
        f"max={alpha_max:.4f} std={alpha_std:.4f} sparse={gate_sparse:.3f}) "
        f"ratio={ratio_mean:.4f} clip={clip_frac:.3f} kl={approx_kl:.4f} ent={dist_entropy:.3f} "
        f"v_loss={value_loss:.4f} a_loss={action_loss:.4f} adv_m={adv_mean:.4f} adv_s={adv_std:.4f} "
        f"ret_m={ret_mean:.4f} val_m={value_pred_mean:.4f} grad={grad_norm_total:.4f} feat_grad={feat_grad_mean:.3e}"
    )


from pathlib import Path
from typing import Optional


def get_run_log_dir(explicit: Optional[str] = None) -> Path:
    """
    依据优先级获取当前 run 的日志目录：
    1. 显式传入
    2. 主 logger 的 FileHandler 目录
    3. 环境变量 FUSION_LOG_DIR
    4. 项目根 logs/
    """
    if explicit:
        return Path(explicit).resolve()

    # 2. root 或 'train' logger 中寻找 FileHandler
    candidates = []
    for name in ("train", "", "root"):
        logger = logging.getLogger(name) if name else logging.getLogger()
        for h in logger.handlers:
            if hasattr(h, "baseFilename"):
                candidates.append(Path(h.baseFilename).parent.resolve())
    if candidates:
        # 取第一个（通常是当前 run 的目录）
        return candidates[0]

    # 3. 环境变量
    import os
    env_dir = os.environ.get("FUSION_LOG_DIR")
    if env_dir:
        return Path(env_dir).resolve()

    # 4. fallback
    return (Path(__file__).resolve().parent.parent.parent / "logs").resolve() 