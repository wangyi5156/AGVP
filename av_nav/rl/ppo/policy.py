#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import abc

import torch
import torch.nn as nn
from torchsummary import summary

from av_nav.common.utils import CategoricalNet
from av_nav.rl.models.rnn_state_encoder import RNNStateEncoder
from av_nav.rl.models.visual_cnn import VisualCNN
from av_nav.rl.models.audio_cnn import AudioCNN

from av_nav.rl.models.fusion.mcan_fusion import MCANFusion
from av_nav.rl.models.fusion.mcan_config import MCANConfig

# from torch.utils.tensorboard import SummaryWriter

DUAL_GOAL_DELIMITER = ','


class Policy(nn.Module):
    def __init__(self, net, dim_actions):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )
        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)
        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


class PointNavMCANNet(nn.Module):
    def __init__(self, observation_space, config, goal_sensor_uuid, extra_rgb=False):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self.config = config
        self._audiogoal = False
        self._pointgoal = False
        self._n_pointgoal = 0

        if ',' in self.goal_sensor_uuid:
            goal1_uuid, goal2_uuid = self.goal_sensor_uuid.split(',')
            self._audiogoal = self._pointgoal = True
            self._n_pointgoal = observation_space.spaces[goal1_uuid].shape[0]
        else:
            if 'pointgoal_with_gps_compass' == self.goal_sensor_uuid:
                self._pointgoal = True
                self._n_pointgoal = observation_space.spaces[self.goal_sensor_uuid].shape[0]
            else:
                self._audiogoal = True

        self.visual_encoder = VisualCNN(observation_space, config.HIDDEN_SIZE, extra_rgb)

        if self._audiogoal:
            if 'audiogoal' in self.goal_sensor_uuid:
                audiogoal_sensor = 'audiogoal'
            elif 'spectrogram' in self.goal_sensor_uuid:
                audiogoal_sensor = 'spectrogram'
            else:
                raise ValueError("Unsupported audio goal sensor.")
            self.audio_encoder = AudioCNN(observation_space, config.HIDDEN_SIZE, audiogoal_sensor)

        self.fusion_module = MCANFusion(config)

        # 新增 GRU 编码器
        self.state_encoder = RNNStateEncoder(
            input_size=config.HIDDEN_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            rnn_type="GRU",
            num_layers=1
        )

        # 模型结构可视化
        if 'rgb' in observation_space.spaces and not extra_rgb:
            rgb_shape = observation_space.spaces['rgb'].shape
            summary(self.visual_encoder.cnn, (rgb_shape[2], rgb_shape[0], rgb_shape[1]), device='cpu')
        if 'depth' in observation_space.spaces:
            depth_shape = observation_space.spaces['depth'].shape
            summary(self.visual_encoder.cnn, (depth_shape[2], depth_shape[0], depth_shape[1]), device='cpu')
        if self._audiogoal:
            audio_shape = observation_space.spaces[audiogoal_sensor].shape
            summary(self.audio_encoder.cnn, (audio_shape[2], audio_shape[0], audio_shape[1]), device='cpu')

        self.train()

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
        audio_feat = self.audio_encoder(observations) if self._audiogoal else None
        visual_feat = self.visual_encoder(observations)
        fused_feat = self.fusion_module(audio_feat, visual_feat)

        x, new_rnn_hidden_states = self.state_encoder(fused_feat, rnn_hidden_states, masks)
        return x, new_rnn_hidden_states


# 这里用新的MCAN策略替代旧的PointNavBaselinePolicy，名字保持不变，不需要改config
class PointNavBaselinePolicy(Policy):
    def __init__(
        self,
        observation_space,
        action_space,
        goal_sensor_uuid,
        hidden_size=512,
        extra_rgb=False
    ):
        mcan_config = MCANConfig(
            HIDDEN_SIZE=hidden_size,
            FUSION_METHOD="sum",  # 也可以换成 "concat" 或其他融合方法
        )

        net = PointNavMCANNet(
            observation_space=observation_space,
            config=mcan_config,
            goal_sensor_uuid=goal_sensor_uuid,
            extra_rgb=extra_rgb
        )

        super().__init__(net=net, dim_actions=action_space.n)
