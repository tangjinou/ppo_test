import gym
import torch
from PPOAgent import PPOAgent
from gym.wrappers import RecordVideo
import argparse
import datetime
import os  # 添加os导入


def train_with_no_ui(num_episodes=3000):
    env = gym.make('CartPole-v1')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    agent = PPOAgent(env, device)
    agent.train(num_episodes)
    
    # 确保checkpoints文件夹存在
    os.makedirs('checkpoints', exist_ok=True)
    # 生成带时间戳的模型文件名，并保存在checkpoints文件夹下
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = os.path.join('checkpoints', f"ppo_model_{timestamp}.pth")
    agent.save_model(model_filename)
    env.close()
    agent.plot_rewards()

def train_with_ui(num_episodes=3000):
    # 创建环境并启用视频录制
    env = gym.make('CartPole-v1', render_mode="human")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    agent = PPOAgent(env, device)
    agent.train(num_episodes)
    
    # 确保checkpoints文件夹存在
    os.makedirs('checkpoints', exist_ok=True)
    # 生成带时间戳的模型文件名，并保存在checkpoints文件夹下
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = os.path.join('checkpoints', f"ppo_model_{timestamp}.pth")
    agent.save_model(model_filename)
    # 训练后进行展示
    print("\n开始展示训练结果...")
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        env.render()  # 渲染当前帧
        action = agent.select_action(state)
        step_result = env.step(action)
        
        if len(step_result) == 5:
            state, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            state, reward, done, _ = step_result
            
        total_reward += reward
    
    print(f"展示完成，总奖励: {total_reward}")
    env.close()
    agent.plot_rewards()


if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='PPO训练程序')
    parser.add_argument('--ui', action='store_true', 
                       help='使用界面模式训练，默认为无界面模式',
                       default=False)
    parser.add_argument('--num_episodes', type=int, default=3000,
                       help='训练轮数，默认为10000')
    
    args = parser.parse_args()

    num_episodes = args.num_episodes

    
    if args.ui:
        train_with_ui(num_episodes)
    else:
        train_with_no_ui(num_episodes)  



