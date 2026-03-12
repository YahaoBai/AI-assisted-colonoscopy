import numpy as np

# 替换为你想要核查的具体文件路径
data = np.load("./dagger_dataset1/step_000008.npz") 

print("专家动作原始张量:", data['action_expert'])
print("策略动作原始张量:", data['action_learner'])
print("总误差原始张量:", data['total_norm_error'])