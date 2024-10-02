# MAPPO算法在多智能体集群围捕场景下的部署和训练
将多智能体强化学习算法（Multi-Agent Reinforcement Learning, MARL）部署到N vs N 多智能体集群围捕任务场景下，进行训练。

## 环境安装
python==3.8
numpy==1.14.5
pytorch==1.10.1
gym==0.10.5
[Multi-Agent Particle-World Environment(MPE)](https://github.com/openai/multiagent-particle-envs)

## 使用
以 MAPPO 为例，通过MAPPO_MPE_main.py 文件进行训练和初步的可视化。
/multiagent 文件夹是目标任务场景：蓝方智能体固定队型，采用人工势场法靠近目标；红方智能体部署MARL算法，通过训练不断优化围捕策略。
训练后的模型/model 将被用于 ROS+Gazebo 环境中的仿真。

## 参考
1) MARL代码框架参考：https://github.com/Lizhi-sjtu/MARL-code-pytorch
2) MAPPO算法：Yu C, Velu A, Vinitsky E, et al. The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games[J]. arXiv preprint arXiv:2103.01955, 2021.