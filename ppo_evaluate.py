import gym
import torch
from PPOAgent import PPOAgent
from gym.wrappers import RecordVideo
import argparse
import datetime
import os  # 添加os导入
import numpy as np  # 在文件开头添加这行



def evaluate_model(model_path, num_episodes=10):  # 增加参数来控制评估次数
    env = gym.make('CartPole-v1')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 添加状态标准化
    state_mean = np.zeros(env.observation_space.shape[0])
    state_std = np.ones(env.observation_space.shape[0])
    
    def normalize_state(state):
        return (state - state_mean) / state_std
    
    agent = PPOAgent(env, device)
    agent.load_model(model_path)
    
    rewards = []  # 存储每次评估的奖励
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            env.render()

            normalized_state = normalize_state(state)  # 标准化状态
            action = agent.select_action_deterministic(normalized_state)  # 使用确定性策略
            
            step_result = env.step(action)
            
            if len(step_result) == 5:
                state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                state, reward, done, _ = step_result
                
            episode_reward += reward
        
        rewards.append(episode_reward)
        print(f"第 {episode + 1} 次评估完成，奖励: {episode_reward}")
    
    rewards = np.array(rewards)  # 将rewards转换为numpy数组
    average_reward = np.mean(rewards)
    std_dev = np.std(rewards)
    
    print(f"\n评估统计:")
    print(f"平均奖励: {average_reward:.2f}")
    print(f"标准差: {std_dev:.2f}")
    print(f"最高奖励: {max(rewards)}")
    print(f"最低奖励: {min(rewards)}")
    
    env.close()

    # 确保evaluate_result文件夹存在
    os.makedirs('evaluate_result', exist_ok=True)
    # 生成带时间戳的评估结果文件名，并保存在evaluate_result文件夹下 
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = os.path.join('evaluate_result', f"evaluate_result_{timestamp}.txt")
    with open(result_filename, 'w') as f:
        f.write(f"模型路径: {model_path}\n")
        f.write(f"评估统计:\n")
        f.write(f"平均奖励: {average_reward:.2f}\n")
        f.write(f"标准差: {std_dev:.2f}\n") 
        f.write(f"最高奖励: {max(rewards)}\n")
        f.write(f"最低奖励: {min(rewards)}\n")
        f.write(f"评估次数: {num_episodes}\n")
        f.write(f"评估时间: {timestamp}\n")



if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='PPO训练程序')

    parser.add_argument('--model_path', type=str, default=None,
                       help='评估模型路径，默认为None')
    parser.add_argument('--num_episodes', type=int, default=500,
                       help='评估次数，默认为500次')  # 添加新的参数

    args = parser.parse_args()

    model_path = args.model_path

    # 如果没有指定模型路径，则获取checkpoints目录下最新的pth文件
    if model_path is None:
        checkpoints_dir = "checkpoints"
        if not os.path.exists(checkpoints_dir):
            print(f"错误：{checkpoints_dir} 目录不存在，请先训练模型")
            exit()
        
        pth_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pth')]
        if not pth_files:
            print(f"错误：在 {checkpoints_dir} 目录中没有找到.pth文件，请先训练模型")
            exit()
            
        latest_file = max(pth_files, key=lambda x: os.path.getctime(os.path.join(checkpoints_dir, x)))
        model_path = os.path.join(checkpoints_dir, latest_file)
        print(f"使用最新的模型文件：{model_path}")

    if not os.path.exists(model_path):
        print(f"错误：模型文件 {model_path} 不存在，请先训练模型")
        exit()

    evaluate_model(model_path, args.num_episodes)  # 传入评估次数参数

