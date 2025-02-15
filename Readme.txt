安装指南：
0 建议用conda创建虚拟环境
conda create --name ppo_env python=3.10
conda activate ppo_env

1. 安装依赖：
pip install -r requirements.txt

2. 运行训练：
python ppo_train.py --ui  #使用界面模式训练，默认为无界面模式

3. 运行评估：
python ppo_evaluate.py --model_path  
#评估模型路径，默认为运行时checkpoints目录下最新的模型文件
#如果为空，则先运行训练python ppo_train.py，再运行评估


其他相关：

## CartPole-v0 环境说明

### 观察空间（Observation Space）
四维连续空间:

| 序号 | 观察值 | 取值范围 | 说明 |
|------|--------|----------|------|
| 0 | 小车位置 | [-4.8, 4.8] | 超出[-2.4, 2.4]时任务终止 |
| 1 | 小车速度 | (-∞, +∞) | - |
| 2 | 杆子角度 | [-0.418, 0.418]弧度 | 超出[-0.2095, 0.2095]弧度时任务终止 |
| 3 | 杆子角速度 | (-∞, +∞) | - |

### 动作空间（Action Space）
离散空间(Discrete):

| 动作值 | 动作含义 |
|--------|----------|
| 0 | 向左推动小车 |
| 1 | 向右推动小车 |

### 环境交互说明
`env.step(action)` 返回四元组 `(observation, reward, done, info)`:

- **observation**: 包含上述4个观察值的数组
- **reward**: 
  - 1.0: 杆子保持直立且小车在边界内
  - 0.0: 失败状态
  - 最大累积奖励: 500
- **done**: 任务结束标志
  - True: 杆子倒下或小车越界
  - False: 继续进行
- **info**: 附加信息（通常为空）

注意：CartPole-v0 的 reward的值 最大的值为500，当然最小才是0