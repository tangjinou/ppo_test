import gym
import numpy as np
from gym import spaces

# 创建一个补货的环境    

class ReplenishEnv(gym.Env):
    def __init__(self, num_stocks=10, max_stock=100, coverage_weight=0.5, rts_weight=0.5):
        super(ReplenishEnv, self).__init__()
        self.num_stocks = num_stocks
        self.max_stock = max_stock
        self.current_stock = np.zeros(num_stocks)
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(num_stocks,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=max_stock, shape=(num_stocks,), dtype=np.float32)
        self.coverage_weight = coverage_weight
        self.rts_weight = rts_weight

    def reset(self):
        self.current_stock = np.random.randint(0, self.max_stock, size=self.num_stocks)
        return self.current_stock

    def step(self, action):
        # 将补货动作限制在0.5到1.5之间
        replenishment_mutiply_rate = np.clip(action, 0.5, 1.5)
        # 将补货量添加到当前库存
        self.current_stock += replenishment_mutiply_rate * self.predicted_demand
        # 确保当前库存不超过最大库存限制
        self.current_stock = np.clip(self.current_stock, 0, self.max_stock)
        
        # 计算 Coverage (覆盖率) = min(当前库存 / 预测需求, 1)
        coverage = np.minimum(self.current_stock / (self.predicted_demand + 1e-8), 1.0)
        coverage_reward = np.mean(coverage)  # 取平均作为覆盖率奖励
        
        # 计算 RTS (库存周转率) = 当前库存 / 预测需求
        rts = self.current_stock / (self.predicted_demand + 1e-8)
        rts_penalty = -np.mean(rts)  # RTS越大，惩罚越大
        
        # 综合奖励：coverage_weight 和 rts_weight 可以根据需要调整
        reward = self.coverage_weight * coverage_reward + self.rts_weight * rts_penalty
        
        # 判断是否达到目标：当所有商品的当前库存等于目标库存时为True
        done = np.all(self.current_stock == self.target_stock)
        # 返回状态时，将当前库存和预测需求连接在一起
        state = np.concatenate([self.current_stock, self.predicted_demand])
        # 返回：当前状态、奖励、是否完成、额外信息（空字典）
        return state, reward, done, {}

    def render(self, mode='human'):
        print(f"当前库存: {self.current_stock}")

if __name__ == "__main__":
    env = ReplenishEnv()
    state = env.reset()
    print(f"初始库存: {state}") 