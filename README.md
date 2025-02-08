# 毕设论文《基于优先级划分的大规模多智能体强化学习决策方法研究》实验部分展示

## 目录
- [实验1-1](#实验1-1-dnf-qmix-qplex-hpnqmix在smacv1上的对比实验)
- [实验1-2](#实验1-2-dnf-dnf-random-dnf-svd-qmix在smacv1上的对比实验)

---

## 实验1-1 (DNF-QMIX-QPLEX-HPNQMIX在SMACv1上的对比实验)

### 实验环境
基准环境为 **[SMACv1](https://github.com/oxwhirl/smac)**（星际争霸多智能体挑战环境）：  
![SMACv1](https://github.com/oxwhirl/smac/blob/master/docs/smac-official.png?raw=true)

---

### 算法框架对比
| 算法名称       | 框架图                                                                               | 参考文献                                              |
|----------------|-----------------------------------------------------------------------------------|---------------------------------------------------|
| **QMIX**       | ![QMIX](https://pic2.zhimg.com/v2-d03d9d93cb31a14a43ff5956528e5159_1440w.jpg)     | [论文链接](https://arxiv.org/abs/1803.11485)          |
| **QPLEX**      | ![QPLEX](https://pica.zhimg.com/v2-f4deda2809e53f4df44e2c08948bc704_1440w.jpg)    | [论文链接](https://arxiv.org/pdf/2008.01062)          |
| **HPN-QMIX**   | ![HPN-QMIX](https://pic3.zhimg.com/v2-5ad94ea8b6195d0563d8d5755b39a2e0_1440w.jpg) | [论文链接](https://openreview.net/pdf?id=OxNQXyZK-K8) |
| **DNF**        | ![DNF](./src/pic/DNF.png)                                                         | 待更新                                               |

---

### 实验结果
#### 多场景对比（SMACv1）
| ![3s_vs_5z](./src/pic/3s_vs_5z.png)   | ![6h_vs_8z](./src/pic/6h_vs_8z.png) | ![5m_vs_6m](./src/pic/5m_vs_6m.png) |
|-------------------------------------|--------------------------------|--------------------------------|
| ![8m_vs_9m](./src/pic/8m_vs_9m.png) | ![2c_vs_64zg](./src/pic/2c_vs_64zg.png) | ![so_many_baneling](./src/pic/so_many_baneling.png) |
| ![MMM2](./src/pic/MMM2.png) | ![corridor](./src/pic/corridor.png) | ![1c3s5z](./src/pic/1c3s5z.png) |

---

### 实验启动方法

```bash
CUDA_VISIBLE_DEVICES="0" nohup python src/main.py --config={algo_name} --env-config=sc2 with env_args.map_name={map_name} obs_agent_id=True obs_last_action=False runner=parallel batch_size_run={parallel_num} buffer_size=5000 t_max={t_max} epsilon_anneal_time=100000 batch_size=128 td_lambda=0.6
```

---

## 实验1-2 (DNF-DNF-Random-DNF-SVD-QMIX在SMACv1上的对比实验)

### 实验结果
| ![3m](./src/pic/3m.png)   | ![8m](./src/pic/8m.png) | ![2m_vs_1z](./src/pic/2m_vs_1z.png) |
|-------------------------------------|--------------------------------|--------------------------------|
| ![2s3z](./src/pic/2s3z.png) | ![3s_vs_4z](./src/pic/3s_vs_4z.png) |  |

---

## 实验2-1 (DNF-DNF-Random-DNF-SVD-QMIX在SMACv1自定义地图上的对比实验)

### 实验结果
| 场景名称         | 算法对比图                    |
|--------------|--------------------------|
| **50m**      | ![图1](./src/pic/50m.png) |

### 实验启动方法

```bash
# 50m run on same configs
CUDA_VISIBLE_DEVICES="0" nohup python src/main.py --config=qmix --env-config=sc2 with env_args.map_name=50m obs_agent_id=True obs_last_action=False runner=parallel batch_size_run=2 buffer_size=5000 t_max=1000000 epsilon_anneal_time=100000 batch_size=128 td_lambda=0.6 > 50m_QPLEX.out &

CUDA_VISIBLE_DEVICES="0" nohup python src/main.py --config=qplex --env-config=sc2 with env_args.map_name=50m obs_agent_id=True obs_last_action=False runner=parallel batch_size_run=2 buffer_size=5000 t_max=1000000 epsilon_anneal_time=100000 batch_size=128 td_lambda=0.6 > 50m_QPLEX.out &

CUDA_VISIBLE_DEVICES="0" nohup python src/main.py --config=hpn_qmix --env-config=sc2 with env_args.map_name=50m obs_agent_id=True obs_last_action=False runner=parallel batch_size_run=2 buffer_size=5000 t_max=1000000 epsilon_anneal_time=100000 batch_size=128 td_lambda=0.6 > 50m_HPN_QMIX.out &

CUDA_VISIBLE_DEVICES="0" nohup python src/main.py --config=dnf --env-config=sc2 with env_args.map_name=50m obs_agent_id=True obs_last_action=False runner=parallel batch_size_run=2 buffer_size=5000 t_max=1000000 epsilon_anneal_time=100000 batch_size=128 td_lambda=0.6 core_extractor_type=nn core_agent_ratio=0.7 > 50m_DNF.out &
```