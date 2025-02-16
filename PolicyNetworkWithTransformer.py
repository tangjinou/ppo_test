import torch
import torch.nn as nn
import torch.distributions as distributions
import math

class PolicyNetworkWithTransformer(nn.Module):
    def __init__(self, state_dim, action_dim=2, hidden_dim=64, num_heads=2, num_layers=1):
        """
        初始化基于Transformer的策略网络
        
        参数:
            state_dim (int): 状态空间维度
            action_dim (int): 动作空间维度，对于CartPole应该是2
            hidden_dim (int): 隐藏层维度，默认为64
            num_heads (int): 注意力头数，默认为2
            num_layers (int): Transformer层数，默认为1
        """
        super(PolicyNetworkWithTransformer, self).__init__()
        
        # 增加输入处理层的深度和宽度
        self.input_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 使用固定的正弦位置编码
        position = torch.arange(1).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * -(math.log(10000.0) / hidden_dim))
        pe = torch.zeros(1, 1, hidden_dim)
        pe[0, 0, 0::2] = torch.sin(position * div_term)
        pe[0, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', pe)
        
        # 添加注意力前的预处理层
        self.pre_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,  # 增大前馈网络维度
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 添加残差连接
        self.residual = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 动作头
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """初始化网络参数"""
        if isinstance(module, nn.Linear):
            # 使用Xavier初始化
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
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
        # 确保输入有批处理维度
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # 保存原始输入用于残差连接
        identity = x
            
        # 输入预处理
        x = self.input_proj(x)
        
        # 添加位置编码
        x = x.unsqueeze(1)
        x = x + self.pos_encoding
        
        # 注意力前的预处理
        x = self.pre_attention(x)
        
        # Transformer处理
        x = self.transformer(x)
        x = x.squeeze(1)
        
        # 添加残差连接
        res = self.residual(identity)
        x = x + res
        
        # 计算动作概率和状态值
        action_logits = self.action_head(x)
        action_probs = torch.softmax(action_logits, dim=-1)
        
        state_value = self.value_head(x)
        
        # 如果输入是单个状态，则移除批处理维度
        if action_probs.size(0) == 1:
            action_probs = action_probs.squeeze(0)
            state_value = state_value.squeeze(0)
        
        return action_probs, state_value