from PolicyNetwork import PolicyNetwork
from PolicyNetworkWithTransformer import PolicyNetworkWithTransformer
from typing import Optional

class PolicyNetworkFactory:
    """策略网络工厂类，用于创建不同类型的策略网络"""
    
    @staticmethod
    def create_policy(network_type: str, state_dim: int, action_dim: int) -> Optional[PolicyNetwork]:
        """
        创建策略网络
        
        参数:
            network_type: 网络类型 ('simple', 'medium', 'large' 等)
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            
        返回:
            PolicyNetwork 实例
        """
        if network_type == "simple":
            return PolicyNetwork(state_dim, action_dim)
        elif network_type == "medium":
            return PolicyNetwork(state_dim, action_dim, hidden_dim=256)
        elif network_type == "large":
            return PolicyNetwork(state_dim, action_dim, hidden_dim=512)
        else:
            raise ValueError(f"未知的网络类型: {network_type}") 