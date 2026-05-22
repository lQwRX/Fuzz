# GANFuzz：基于生成对抗网络的Modbus TCP协议模糊测试系统

本系统实现了论文《GANFuzz: A GAN-based industrial network protocol fuzzing》的核心方法，并扩展了三种聚类策略、闭环反馈优化和基线对比实验。系统能够在不依赖协议规范的前提下，自动学习Modbus TCP协议语法，生成高质量测试用例，并对目标从站进行模糊测试。

## 目录结构及功能说明
```
Fuzz/
├── capture/ # 消息捕获与分析模块（MCAM）
│ ├── init.py
│ └── message_collector.py # 从pcap提取Modbus消息或生成模拟数据
│
├── preprocessing/ # 预处理与聚类模块（MPM）
│ ├── clustering/ # 三种聚类策略实现
│ │ ├── advanced_clustering.py # 高级聚类（k‑means）
│ │ ├── no_clustering.py # 无聚类
│ │ ├── same_length_clustering.py # 等长聚类
│ │ └── base.py # 聚类公用函数
│ ├── cleaner.py # 清洗（长度过滤、去重）
│ ├── numericalizer.py # 十六进制→整数序列
│ ├── padder.py # 序列填充/截断
│ └── vocab_builder.py # 构建字节到索引的词汇表
│
├── model/ # SeqGAN 模型与训练模块（TMGM）
│ ├── generator.py # 生成器网络（LSTM）
│ ├── discriminator.py # 判别器网络（CNN）
│ ├── dataset.py # PyTorch Dataset封装
│ ├── pretrain.py # 预训练生成器和判别器
│ ├── adversarial.py # 对抗交替训练（策略梯度+蒙特卡洛）
│ ├── rollout.py # 蒙特卡洛搜索计算奖励
│ ├── train_cluster.py # 训练单个聚类子集
│ ├── multi_cluster_trainer.py # 训练所有聚类子集的调度器
│ ├── generate_cases.py # 从训练好的模型生成测试用例
│ ├── retrain.py # 闭环反馈微调（使用异常用例重训练生成器）
│ └── checkpoints_improved/ # 保存的训练模型（按聚类/子类存放）
│
├── fuzzer/ # 模糊测试执行模块（MSM/MM/LM）
│ ├── sender.py # TCP发送器，连接从站发送十六进制消息
│ ├── logger.py # 日志记录器，写入请求/响应/状态
│ └── fuzzer_main.py # 模糊测试主入口，支持指定测试用例文件
│
├── baseline/ # 传统基线方法（对比实验用）
│ ├── random_generator.py # 完全随机字节序列生成器
│ └── mutation_generator.py # 基于真实种子的随机变异生成器
│
├── evaluation/ # 评估指标计算模块
│ └── metrics.py # 计算 TIRR（测试输入拒绝率）和 AoVD（每百用例异常数）
│
├── experiments/ # 对比实验与结果可视化
│ ├── run_compare.py # 运行随机/变异/GAN三种方法并计算指标
│ └── plot_results.py # 绘制 TIRR/AoVD 对比柱状图
│
├── utils/ # 通用工具函数
│ ├── file_utils.py # 文件读写、目录创建、JSON保存
│ └── conversion_utils.py # 十六进制↔字节↔整数序列转换
│
├── data/ # 数据目录（自动生成）
│ ├── raw_messages.txt # 原始十六进制消息（来自 pcap 和模拟混合）
│ ├── raw_messages.txt #来自 pcap 的十六进制消息
│ └── clusters/ # 聚类后的数值数据子集
│   ├── no_clustering/ # 无聚类结果（直接存放 data.npy, vocab.json, max_len.txt）
│   ├── same_length/ # 等长聚类子目录（class_0, class_1, ...）
│   └── advanced/ # 高级聚类子目录（class_0 ~ class_7）
│
├── pcap/ # 存放原始 Modbus TCP 抓包文件（可选）
│
├── output/ # 运行输出目录（自动创建）
│ ├── models/ # 训练时保存的模型（checkpoints_improved 的副本或训练输出）
│ ├── test_cases/ # 生成的测试用例（gan_cases.txt 及分聚类文件）
│ ├── logs/ # 模糊测试日志（baseline_random.log, baseline_mutation.log, fuzz_gan.log）
│ ├── reports/ # 评估报告（comparison.json）
│ └── figures/ # 可视化图表（comparison.png）
│
├── config.py # 全局配置文件（路径、超参数、聚类参数、测试配置）
├── train_all.py # 主训练入口（一键执行预处理与所有聚类训练）
├── requirements.txt # Python 依赖清单
└── README.md # 本说明文件
```
## 快速开始
```bash
### 1. 环境准备

- Python 3.8 或 3.9（建议使用虚拟环境）
- 安装依赖：`pip install -r requirements.txt`

### 2. 数据准备

- 将 Modbus TCP 的 pcap 文件放入 `pcap/` 目录（可选，若没有则自动生成模拟数据）
- 运行数据预处理与聚类：`python script/run_preprocessing.py`  
  （该脚本会生成 `data/clusters/` 下的所有子集）

### 3. 模型训练

- 运行训练脚本：`python train_all.py`  
  系统会自动遍历 `data/clusters/` 中的每个聚类子集，依次完成预训练和10步对抗训练，每5步保存一次模型到 `model/checkpoints_improved/`。

### 4. 生成测试用例

- 运行：`python model/generate_cases.py`  
  从各聚类子集的 step10 生成器中采样测试用例，合并后保存到 `output/test_cases/gan_cases.txt`，同时按聚类分别保存。

### 5. 启动 Modbus 从站

- 推荐使用 `diagslave`：`diagslave -m tcp -p 502`  
- 或使用 Modbus Slave 图形软件，配置 TCP/IP 模式，端口 502。

### 6. 执行模糊测试与对比

- 运行对比实验：`python experiments/run_compare.py`  
  该脚本依次执行随机生成、随机变异、GAN 测试，计算 TIRR 和 AoVD，并将结果保存到 `output/reports/comparison.json`。

### 7. 结果可视化

- 运行绘图：`python experiments/plot_results.py`  
  生成柱状图 `output/figures/comparison.png`。

### 8. 闭环反馈微调

- 在获得测试日志后（如 `output/logs/fuzz_gan.log`），可运行：  
  `python model/retrain.py --log output/logs/fuzz_gan.log --cluster no_clustering --class_idx 0 --epochs 2`  
  该脚本会提取异常用例，突变后与原始训练数据合并，对生成器进行微调，并将新模型保存至 `model/checkpoints_improved/no_clustering/class_0/retrain/`。
```
## 主要算法与指标

- **生成器**：LSTM（隐藏单元32，嵌入维度256）
- **判别器**：多尺度 CNN（卷积核尺寸 2,3,4,5，各100个滤波器）
- **训练策略**：预训练（教师强制）→ 判别器预训练 → 对抗交替训练（策略梯度 + 蒙特卡洛搜索，rollout=5，10个交替步）
- **评价指标**：
  - **TIRR**（测试输入拒绝率）= 被拒绝用例数 / 总用例数 × 100%（越低越好）
  - **AoVD**（每百用例异常数）= 异常用例数 / 总用例数 × 100（越高越好）

## 依赖库

- numpy, scikit-learn, scapy, torch, tqdm, matplotlib

## 参考文献

Chu Z, Shi J, Huang Y, et al. GANFuzz: A GAN-based industrial network protocol fuzzing[C]//Proceedings of the 15th ACM International Conference on Computing Frontiers. ACM, 2018: 1-8.