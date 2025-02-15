import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PolicyNetwork import PolicyNetwork

class PPOMemory:
    def __init__(self, batch_size=32):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.batch_size = batch_size
        
    def push(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        
    def sample(self):
        # 随机采样batch_size大小的经验
        indices = np.random.choice(len(self.states), self.batch_size)
        return (
            torch.FloatTensor(np.array(self.states)[indices]),
            torch.FloatTensor(np.array(self.actions)[indices]),
            torch.FloatTensor(np.array(self.rewards)[indices]),
            torch.FloatTensor(np.array(self.next_states)[indices]),
            torch.FloatTensor(np.array(self.dones)[indices])
        )

class PPOAgent:
    def __init__(self, env, device):
        """
        初始化 PPO 智能体
        
        参数:
            env: OpenAI Gym 环境
            device: 计算设备（CPU/GPU）
        """
        self.env = env
        self.device = device
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.policy = PolicyNetwork(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.0001)
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.actor_lr = 0.0001
        self.critic_lr = 0.0001
        self.eps_clip = 0.2
        self.K_epochs = 10
        
        # 存储训练数据
        self.states = []
        self.actions = []
        self.logprobs = []
        self.state_values = []
        self.rewards = []
        self.dones = []
        self.episode_rewards = []

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
        for _ in range(self.K_epochs):
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
        """绘制奖励趋势图"""
        data = np.array(self.episode_rewards)
        plt.figure(figsize=(12, 6))
        plt.plot(data, linewidth=1)
        plt.title('奖励趋势')
        plt.xlabel('回合数')
        plt.ylabel('奖励值')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

    def train(self, max_episodes):
        print(f"开始训练 \t 训练轮数: {max_episodes}")

        """训练智能体"""
        best_reward = 0
        stable_count = 0
        patience = 50  # 连续50轮都保持高分才保存模型
        
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
            
            if (episode + 1) % 10 == 0:
                print(f"回合 {episode + 1}\t总奖励: {total_reward}")

            if total_reward >= 495:  # 接近最大奖励
                stable_count += 1
            else:
                stable_count = 0
            
            if stable_count >= patience and total_reward > best_reward:
                best_reward = total_reward
                self.save_model(f"best_model_{best_reward:.0f}.pth")
            
            # 如果连续100轮都达到接近最大奖励，可以提前结束
            if stable_count >= 100:
                print(f"提前结束训练：已经连续{stable_count}轮达到高分")
                break

    def evaluate(self, state):
        """评估状态"""
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            probs, state_value = self.policy(state)
        return probs, state_value

    def save_model(self, path):
        """保存模型"""
        torch.save(self.policy.state_dict(), path)

    def load_model(self, path):
        """加载模型"""
        self.policy.load_state_dict(torch.load(path))
        self.policy.to(self.device) 