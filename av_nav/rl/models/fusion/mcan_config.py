from dataclasses import dataclass


@dataclass
class MCANConfig:
    """
    Configuration class for MCAN encoder-decoder fusion module.
    Can be loaded via Hydra or argparse.
    """
    HIDDEN_SIZE: int = 256                 # Feature dimension D
    MULTI_HEAD: int = 4                    # Number of attention heads
    FF_SIZE: int = 1024                    # Feedforward hidden size
    DROPOUT_R: float = 0.6                 # Dropout rate
    LAYER: int = 2                         # Number of encoder/decoder layers
    FUSION_METHOD: str = "sum"            # Options: sum, concat, visual_only, audio_only

    @property
    def HIDDEN_SIZE_HEAD(self):
        return self.HIDDEN_SIZE // self.MULTI_HEAD

#config = MCANConfig(HIDDEN_SIZE=512, LAYER=4, FUSION_METHOD='concat')   
"""
HIDDEN_SIZ:每个 token（视觉 patch 或音频帧）表示的维度 D.
    控制特征维度，是所有注意力层、FFN 层的输入/输出维度。数值越大，模型表达力越强，但计算量也越大

LAYER=4:encoder 和 decoder 的层数（即：SA × 4 + SGA × 4）

FUSION_METHOD='concat':最后将音频向量 a 和视觉向量 v 如何融合成一个向量 z.
'concat' 表示 [v ; a] → FC → z，保留两种模态独立信息，比 sum 更灵活但参数更多
"""

"""
使用情形	建议参数
快速调试	LAYER=1~2, FUSION_METHOD='sum'
强融合学习	LAYER=4, FUSION_METHOD='concat'
小模型部署	HIDDEN_SIZE=256, LAYER=2
泛化/鲁棒性测试	FUSION_METHOD='visual_only' 或 'audio_only'
"""
