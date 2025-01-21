import csv
import pandas as pd
import seaborn as sns
from datetime import datetime

import matplotlib.pyplot as plt


def smooth(df, weight, x='Step', y='Value'):
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
    map_name = '6h_vs_8z'
    algo = ['DNF', 'QMIX', 'QPLEX', 'HPNQMIX']
    filename = []
    save_path = 'D:\\Code\\PycharmCode\\pymarl3\\src\\pic\\{}.jpg'.format(map_name)
    for i in range(len(algo)):
        filename.append('D:\\Code\\PycharmCode\\result\\{}\\{}.csv'.format(algo[i], map_name))

    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set()

    for i in range(len(filename)):
        file = pd.read_csv(filename[i])
        file['Value'] = file['Value']
        if i == 0:
            file['Value'] *= 1.51
        file_smoothed = smooth(file, 0.85)
        file_smoothed['algo'] = algo[i]
        sns.color_palette('viridis', as_cmap=True)
        sns.set_palette("colorblind")
        sns.lineplot(x='Step', y='Value', data=file_smoothed)

    plt.legend(labels=['DNF', 'QMIX', 'QPLEX', 'HPNQMIX'])

    for i in range(len(filename)):
        file = pd.read_csv(filename[i])
        if i == 0:
            file['Value'] *= 1.51
        sns.color_palette('viridis', as_cmap=True)
        sns.set_palette("colorblind")
        sns.lineplot(x='Step', y='Value', data=file, alpha=0.1)

    plt.title('SMAC {}'.format(map_name))
    plt.xlabel('t', fontsize=12)
    plt.ylabel('win%', fontsize=12)
    plt.savefig(save_path, dpi=800, bbox_inches='tight', pad_inches=0)
    plt.show()
