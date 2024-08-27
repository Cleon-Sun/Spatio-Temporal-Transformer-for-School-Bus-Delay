import torch
from helper import get_data_train_no_X, import_data_new, get_adj
from model.STTN import STTN
import numpy as np

file_path = "school_weekly_weather_traffic2(1).csv"

df = import_data_new(file_path, train_test='train', weather='No',traffic='No',season_dummy='No')
train = get_data_train_no_X(df, target_name='Avg_Delay_Time')
mean = torch.mean(train).item()
std = torch.std(train).item()


df_test = import_data_new(file_path, train_test='test', weather='No',traffic='No',season_dummy='No')
origin_test = get_data_train_no_X(df_test, target_name='Avg_Delay_Time')
test = (origin_test - mean) / std

A = get_adj("adjacency_matrixdjace.npy")
A_wave = A + np.eye(A.shape[0])
D_wave = np.eye(A.shape[0]) * (np.sum(A_wave, axis=0) ** (-0.5))
adj = np.matmul(np.matmul(D_wave, A_wave), D_wave)
adj = adj[:, :32]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = STTN(adj=adj, len_his=3, len_pred=1).to(device)
model.load_state_dict(torch.load("checkpoint/STTN.pkl", map_location=device))
model.eval()
# 计算每周所有学校的平均延迟时间
avg_delay_per_week = torch.mean(origin_test, dim=1).squeeze(-1)

predicted_delays = []

# 遍历所有周数，从第4周开始
for week_start in range(0, test.shape[0] - 3):
    input_data = test[week_start:week_start+3].unsqueeze(0).to(device)
    with torch.no_grad():
        predicted = (model(input_data) * std + mean).squeeze(0)
    # 储存预测的结果
    predicted_delays.append(torch.mean(predicted, dim=1).squeeze(-1).cpu().numpy())

# 将预测的结果从列表中转化为tensor
predicted_delays_tensor = torch.tensor(predicted_delays)

import matplotlib.pyplot as plt

# 绘制实际的平均延迟时间
weeks = list(range(1, avg_delay_per_week.shape[0] + 1))
plt.plot(weeks, avg_delay_per_week.numpy(), label='Actual Delay')

# 绘制预测的延迟时间
predicted_weeks = list(range(4, avg_delay_per_week.shape[0] + 1))
plt.plot(predicted_weeks, predicted_delays_tensor.numpy(), linestyle='--', label='Predicted Delay')

# 添加标题和标签
plt.title("Average Delay per Week")
plt.xlabel("Week")
plt.ylabel("Average Delay")
plt.legend()
plt.show()

