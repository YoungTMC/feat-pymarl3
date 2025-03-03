import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def smooth(df, weight, x='Step', y='Value'):
    """平滑数据函数"""
    scalar = df[y].values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_value = last * weight + (1 - weight) * point
        smoothed.append(smoothed_value)
        last = smoothed_value

    smoothed_df = pd.DataFrame({x: df[x].values, y: smoothed})
    return smoothed_df

if __name__ == '__main__':
    # 实验设置
    map_name = '3s_vs_5z'
    algo = ['DNF', 'VDN', 'QMIX', 'QPLEX', 'HPNQMIX']
    
    # 自定义颜色
    custom_colors = ['#1072BD', '#77AE43', '#EDB021', '#D7592C', '#7F318D']
    
    # 文件路径
    filename = []
    for alg in algo:
        filename.append(f'../result/{alg}/{map_name}.csv')
    
    # 保存路径
    save_path = f'./pic/{map_name}_comparison.png'

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 设置字体
    plt.rcParams['font.family'] = ['serif', 'sans-serif']
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    # 设置绘图风格
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    plt.rcParams['axes.unicode_minus'] = False    # 用于显示负号
    
    # 绘制平滑曲线
    for i, file_path in enumerate(filename):
        try:
            # 读取CSV文件
            file = pd.read_csv(file_path)
            
            # 平滑数据
            smooth_weight = 0.85
            file_smoothed = smooth(file, smooth_weight)
            file_smoothed['algo'] = algo[i]
            
            # 绘制平滑曲线
            plt.plot(file_smoothed['Step'], file_smoothed['Value'], 
                     color=custom_colors[i], linewidth=2, label=algo[i])
            
            # 绘制原始数据（透明度低）
            plt.plot(file['Step'], file['Value'], color=custom_colors[i], 
                     alpha=0.1, linewidth=1)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # 添加图例和标签
    plt.legend(loc='best')
    plt.title(f'{map_name}', fontsize=14)
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('Win Rate (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 设置y轴范围（可选，根据数据调整）
    # plt.ylim(0, 1.0)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
