import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PolicyNetwork import PolicyNetwork
from PolicyNetworkFactory import PolicyNetworkFactory

class PPOAgent:
    def __init__(self, env, device, network_type="simple", early_stop=False, lr=3e-4, gamma=0.99, k_epochs=4, eps_clip=0.2):
        """
        初始化 PPO 智能体
        
        参数:
            env: OpenAI Gym 环境
            device: 计算设备（CPU/GPU）
            network_type: 策略网络类型 ('simple', 'medium', 'large')
            lr: 学习率
            gamma: 折扣因子
            k_epochs: PPO更新轮数
            eps_clip: PPO裁剪参数
        """
        self.env = env
        self.device = device
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        print(f"PPOAgent init: network_type = {network_type}")

        # 使用工厂类创建策略网络
        self.policy = PolicyNetworkFactory.create_policy(
            network_type, 
            self.state_dim, 
            self.action_dim
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.k_epochs = k_epochs
        self.eps_clip = eps_clip
        
        # 存储训练数据
        self.states = []
        self.actions = []
        self.logprobs = []
        self.state_values = []
        self.rewards = []
        self.dones = []
        self.episode_rewards = []

        # 添加 early stopping 相关参数
        self.early_stop = early_stop
        self.early_stop_patience = 100  # 容忍多少次没有改善
        self.early_stop_window = 20    # 计算平均奖励的窗口大小
        self.best_avg_reward = float('-inf')
        self.no_improve_count = 0

        # 保存网络类型
        self.network_type = network_type

    def select_action(self, state):
        """选择动作"""
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        probs, state_value = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        
        self.states.append(state.detach())
        self.actions.append(action.detach())
        self.logprobs.append(m.log_prob(action).detach())
        self.state_values.append(state_value.detach())
        
        return action.item()

    def select_action_deterministic(self, state):
        """确定性地选择动作"""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            probs, state_value = self.policy(state)  # 解包返回值，只使用概率
            action = torch.argmax(probs).item()
        return action

    def update(self):
        """更新策略网络"""
        states = torch.stack(self.states).to(self.device)
        actions = torch.tensor(self.actions, dtype=torch.int64).to(self.device)
        logprobs = torch.stack(self.logprobs).to(self.device)
        state_values = torch.cat(self.state_values).squeeze().to(self.device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(self.dones, dtype=torch.float32).to(self.device)

        # 计算回报和优势值
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = returns - state_values.detach()

        # 标准化优势值
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO更新
        for _ in range(self.k_epochs):
            probs, state_values_new = self.policy(states)
            m = Categorical(probs)
            new_logprobs = m.log_prob(actions)
            entropy = m.entropy().mean()

            ratios = torch.exp(new_logprobs - logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2).mean() + \
                   0.5 * nn.MSELoss()(state_values_new.squeeze(), returns) - \
                   0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 清空缓存
        self.states = []
        self.actions = []
        self.logprobs = []
        self.state_values = []
        self.rewards = []
        self.dones = []

    def plot_rewards(self):
        """绘制奖励趋势图并保存"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统
        # plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        data = np.array(self.episode_rewards)
        plt.figure(figsize=(12, 6))
        plt.plot(data, linewidth=1)
        
        # 使用英文标题避免字体问题
        plt.title('Reward Trend with network type: ' + self.network_type)
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 生成时间戳文件名并保存图片
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoints/reward_trend_{timestamp}.jpg"
        plt.savefig(filename)
        plt.close()

    def train(self, max_episodes=3000):
        """训练智能体"""
        for episode in range(max_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action = self.select_action(state)
                step_result = self.env.step(action)

                if len(step_result) == 5:
                    next_state, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    next_state, reward, done, _ = step_result

                self.rewards.append(reward)
                self.dones.append(done)
                state = next_state
                total_reward += reward
                
            self.update()
            self.episode_rewards.append(total_reward)
            
            # Early stopping 检查
            if self.early_stop == "true" and episode >= self.early_stop_window:
                recent_rewards = self.episode_rewards[-self.early_stop_window:]
            # early_stop 的代码待实现 还没有想清楚
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-self.early_stop_window:]) if episode >= self.early_stop_window else np.mean(self.episode_rewards)
                print(f"回合 {episode + 1}\t总奖励: {total_reward:.2f}\t最近{self.early_stop_window}回合平均奖励: {avg_reward:.2f}")

    def evaluate(self, state):
        """评估状态"""
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            probs, state_value = self.policy(state)
        return probs, state_value

    def save_model(self, path):
        """保存模型和网络配置"""
        model_info = {
            'state_dict': self.policy.state_dict(),
            'network_type': self.network_type,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }
        torch.save(model_info, path)

    def load_model(self, path):
        """加载模型和网络配置"""
        print(f"加载模型: {path}")
        model_info = torch.load(path)
        print(model_info)
        # 使用保存的配置重新创建策略网络
        self.policy = PolicyNetworkFactory.create_policy(
            model_info['network_type'],
            model_info['state_dim'],
            model_info['action_dim']
        ).to(self.device)
        
        # 加载模型参数
        self.policy.load_state_dict(model_info['state_dict'])
        self.policy.to(self.device)

        return model_info 