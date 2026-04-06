#!/usr/bin/env python3

from av_nav.common.baseline_registry import baseline_registry
from av_nav.rl.ppo.policy import Policy
from av_nav.rl.models.fusion.mcan_fusion import MCANFusion
from av_nav.rl.models.fusion.mcan_config import MCANConfig
from av_nav.rl.models.visual_cnn import VisualCNN
from av_nav.rl.models.audio_cnn import AudioCNN
from av_nav.rl.models.rnn_state_encoder import RNNStateEncoder

import torch.nn as nn

class PointNavMCANNet(nn.Module):
    def __init__(self, observation_space, config, goal_sensor_uuid, extra_rgb=False):
        super().__init__()

        self.goal_sensor_uuid = goal_sensor_uuid
        self.config = config

        self._audiogoal = "audiogoal" in goal_sensor_uuid or "spectrogram" in goal_sensor_uuid
        self._pointgoal = "pointgoal" in goal_sensor_uuid

        # 编码器
        self.visual_encoder = VisualCNN(observation_space, config.HIDDEN_SIZE, extra_rgb)
        if self._audiogoal:
            self.audio_encoder = AudioCNN(observation_space, config.HIDDEN_SIZE, goal_sensor_uuid)

        # 融合器
        self.fusion_module = MCANFusion(config)

        # RNN 状态编码器（使用官方定义）
        self.state_encoder = RNNStateEncoder(
            input_size=config.HIDDEN_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            rnn_type="GRU",
            num_layers=1,
        )

    @property
    def output_size(self):
        return self.config.HIDDEN_SIZE

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    def forward(self, observations, rnn_hidden_states, prev_actions=None, masks=None):
        visual_feat = self.visual_encoder(observations)
        audio_feat = self.audio_encoder(observations) if self._audiogoal else None

        fused_feat = self.fusion_module(audio_feat, visual_feat)

        # 输入到 RNNStateEncoder 中（统一处理单步和序列）
        x, new_rnn_hidden_states = self.state_encoder(fused_feat, rnn_hidden_states, masks)

        return x, new_rnn_hidden_states