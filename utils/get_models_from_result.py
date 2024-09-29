import os
import pickle
import matplotlib.pyplot as plt
from enums.enums import Data, Rmse
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(MLP, self).__init__()
        layers = []
        in_size = input_size
        for h in hidden_layers:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def load_rmse_from_pkl(folder_path):
    rmse_values = {}

    # 폴더 내 모든 파일을 순회
    for filename in os.listdir(folder_path):
        if filename.endswith(".pkl"):
            file_path = os.path.join(folder_path, filename)

            # pkl 파일 불러오기
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                rmse_values[filename] = data[Rmse.BEST_RMSE]

    return rmse_values

def plot_rmse(rmse_dict):
    # 파일 이름과 RMSE 값 분리
    files = list(rmse_dict.keys())
    rmse_values = list(rmse_dict.values())

    # 그래프 그리기
    plt.figure(figsize=(10, 6))
    plt.bar(files, rmse_values, color='skyblue')
    plt.xlabel('File Name', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('RMSE Values from PKL Files', fontsize=14)

    # x축 파일 이름이 잘 보이게 회전
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.show()