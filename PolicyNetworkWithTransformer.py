import torch
import torch.nn as nn
import torch.distributions as distributions

class PolicyNetworkWithTransformer(nn.Module):
    def __init__(self, state_dim, action_dim=2, hidden_dim=64, num_heads=4, num_layers=2):
        """
        初始化基于Transformer的策略网络
        
        参数:
            state_dim (int): 状态空间维度
            action_dim (int): 动作空间维度，对于CartPole应该是2
            hidden_dim (int): 隐藏层维度，默认为64
            num_heads (int): 注意力头数，默认为4
            num_layers (int): Transformer层数，默认为2
        """
        super(PolicyNetworkWithTransformer, self).__init__()
        
        # 输入投影层
        self.input_proj = nn.Linear(state_dim, hidden_dim)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出头 - 确保动作维度为2（CartPole的动作空间）
        self.action_head = nn.Linear(hidden_dim, 2)  # 固定为2个动作
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x (torch.Tensor): 输入状态张量，形状为 (batch_size, state_dim) 或 (state_dim,)
            
        返回:
            tuple: (action_probs, state_value)
                - action_probs: 动作概率分布
                - state_value: 状态值估计（标量）
        """
        # 确保输入有批处理维度
        if x.dim() == 1:
            x = x.unsqueeze(0)  # 添加批处理维度
            
        # 将输入投影到隐藏维度
        x = self.input_proj(x)
        
        # 添加位置编码
        x = x.unsqueeze(1)  # 添加序列维度
        x = x + self.pos_encoding
        
        # Transformer处理
        x = self.transformer(x)
        x = x.squeeze(1)  # 移除序列维度
        
        # 计算动作概率分布（只有两个动作：0和1）
        action_logits = self.action_head(x)
        action_probs = torch.softmax(action_logits, dim=-1)
        
        # 计算状态值
        state_value = self.value_head(x)
        
        # 如果输入是单个状态（没有批处理维度），则移除批处理维度
        if action_probs.size(0) == 1:
            action_probs = action_probs.squeeze(0)
            state_value = state_value.squeeze(0)
        
        return action_probs, state_value