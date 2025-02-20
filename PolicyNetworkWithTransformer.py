import torch
import torch.nn as nn
import torch.distributions as distributions
import math

class PolicyNetworkWithTransformer(nn.Module):
    def __init__(self, state_dim, action_dim=2, hidden_dim=64, num_heads=2, num_layers=1, history_length=2):
        """
        初始化基于Transformer的策略网络
        
        参数:
            state_dim (int): 状态空间维度
            action_dim (int): 动作空间维度，对于CartPole应该是2
            hidden_dim (int): 隐藏层维度，默认为64
            num_heads (int): 注意力头数，默认为2
            num_layers (int): Transformer层数，默认为1
            history_length (int): 历史状态长度，默认为2
        """
        super(PolicyNetworkWithTransformer, self).__init__()
        
        # 增加输入处理的表达能力
        self.input_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 使用固定的正弦位置编码
        position = torch.arange(history_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * -(math.log(10000.0) / hidden_dim))
        pe = torch.zeros(1, history_length, hidden_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', pe)
        
        # 增强Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,  # 增大前馈网络维度
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 增强动作头
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 增强价值头
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 温度参数
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        
        self.history_length = history_length
        self.state_history = None
        
        # 使用正交初始化
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """初始化网络参数"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)  # 增大gain值
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
                
    def reset_history(self):
        self.state_history = None
        
    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x (torch.Tensor): 输入状态张量，形状为 (batch_size, state_dim) 或 (state_dim,)
            
        返回:
            tuple: (action_probs, state_value)
                - action_probs: 动作概率分布
                - state_value: 状态值估计
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        batch_size = x.size(0)
        
        # 更新历史状态
        if self.state_history is None:
            self.state_history = x.unsqueeze(1).repeat(1, self.history_length, 1)
        else:
            if self.state_history.size(0) != batch_size:
                self.state_history = x.unsqueeze(1).repeat(1, self.history_length, 1)
            else:
                self.state_history = torch.cat([
                    self.state_history[:, 1:],
                    x.unsqueeze(1)
                ], dim=1)
        
        # 特征提取
        x = self.input_proj(self.state_history)
        x = x + self.pos_encoding
        
        # Transformer处理
        x = self.transformer(x)
        x = torch.mean(x, dim=1)  # 使用平均池化而不是只取最后一个时间步
        
        # 计算动作概率和状态值
        action_logits = self.action_head(x)
        temperature = torch.clamp(self.temperature, min=0.1, max=1.0)  # 降低温度上限
        action_probs = torch.softmax(action_logits / temperature, dim=-1)
        
        state_value = self.value_head(x)
        
        if action_probs.size(0) == 1:
            action_probs = action_probs.squeeze(0)
            state_value = state_value.squeeze(0)
        
        return action_probs, state_value