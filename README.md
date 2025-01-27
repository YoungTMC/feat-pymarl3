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
| 算法名称       | 框架图                                                                 | 参考文献                                                                 |
|----------------|------------------------------------------------------------------------|--------------------------------------------------------------------------|
| **QMIX**       | ![QMIX](https://pic2.zhimg.com/v2-d03d9d93cb31a14a43ff5956528e5159_1440w.jpg) | [论文链接](https://arxiv.org/abs/1803.11485)                             |
| **QPLEX**      | ![QPLEX](https://pica.zhimg.com/v2-f4deda2809e53f4df44e2c08948bc704_1440w.jpg) | [论文链接](https://arxiv.org/pdf/2008.01062)                             |
| **HPN-QMIX**   | ![HPN-QMIX](https://pic3.zhimg.com/v2-5ad94ea8b6195d0563d8d5755b39a2e0_1440w.jpg) | [论文链接](https://openreview.net/pdf?id=OxNQXyZK-K8)                    |
| **DNF**        | ![DNF](D:\Study\硕士毕设材料\DNF.png)                                  | [本地文档](D:\Study\硕士毕设材料\ICCC 2024\第三稿.docm)                  |

---

### 实验结果
#### 多场景对比（SMACv1）
| 场景名称              | 算法对比图                                                  |
|-----------------------|--------------------------------------------------------|
| **3s_vs_5z**         | ![3s_vs_5z](./src/pic/3s_vs_5z.png)                    |
| **6h_vs_8z**         | ![6h_vs_8z](./src/pic/6h_vs_8z.png)                    |
| **5m_vs_6m**         | ![5m_vs_6m](./src/pic/5m_vs_6m.png)                    |
| **8m_vs_9m**         | ![8m_vs_9m](./src/pic/8m_vs_9m.png)                    |
| **2c_vs_64zg**       | ![2c_vs_64zg](./src/pic/2c_vs_64zg.png)                |
| **so_many_baneling** | ![so_many_baneling](./src/pic/so_many_baneling.png) |
| **MMM2**             | ![MMM2](./src/pic/MMM2.png)             |
| **corridor**         | ![corridor](./src/pic/corridor.png)         |
| **1c3s5z**           | ![1c3s5z](./src/pic/1c3s5z.png)           |

---

## 实验1-2 (DNF-DNF-Random-DNF-SVD-QMIX在SMACv1上的对比实验)

### 实验结果
| 场景名称      | 算法对比图                    |
|---------------|------------------------------|
| **3m**       | ![3m](./src/pic/3m.png) |
| **8m**       | ![8m](/src/pic/8m.png) |
| **2m_vs_1z** | ![2m_vs_1z](/src/pic/2m_vs_1z.png) |
| **2s3z**     | ![2s3z](/src/pic/2s3z.png) |
| **3s_vs_4z** | ![3s_vs_4z](/src/pic/3s_vs_4z.png) |

---

## 实验2-1 (DNF-DNF-Random-DNF-SVD-QMIX在SMACv1自定义地图上的对比实验)

### 实验结果
| 场景名称         | 算法对比图                    |
|--------------|------------------------------|
| **50m**      | ![图1](./src/pic/3m.png) |