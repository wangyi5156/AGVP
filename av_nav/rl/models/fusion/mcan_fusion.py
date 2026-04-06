import torch
import torch.nn as nn
from av_nav.rl.models.fusion.mca import MCA_ED
from av_nav.rl.models.fusion.net_utils import LayerNorm, MLP


class MCANFusion(nn.Module):
    """
    MCAN-based Encoder-Decoder fusion module
    - Audio feature is the encoder input (Y)
    - Visual feature is the decoder input (X)
    Output: fused feature [B, D] for policy input
    """
    def __init__(self, config, use_audio_output=True, fusion_method="sum"):
        super().__init__()
        self.config = config
        self.mca_ed = MCA_ED(config)
        self.norm = LayerNorm(config.HIDDEN_SIZE)
        self.use_audio_output = use_audio_output
        self.fusion_method = fusion_method.lower()

    def forward(self, audio_feat, visual_feat, audio_mask=None, visual_mask=None):
        # audio_feat: [B, T, D] - Encoder input
        # visual_feat: [B, N, D] - Decoder input

        audio_ctx, visual_ctx = self.mca_ed(audio_feat, visual_feat, audio_mask, visual_mask)

        # Reduce sequence to vector
        audio_vec = audio_ctx.mean(dim=1)  # [B, D]
        visual_vec = visual_ctx.mean(dim=1)  # [B, D]

        if self.fusion_method == "sum":
            fused = self.norm(audio_vec + visual_vec)
        elif self.fusion_method == "concat":
            fused = self.proj(torch.cat([visual_vec, audio_vec], dim=1))  # [B, D]
        elif self.fusion_method == "visual_only":
            fused = visual_vec
        elif self.fusion_method == "audio_only":
            fused = audio_vec
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")

        return fused
