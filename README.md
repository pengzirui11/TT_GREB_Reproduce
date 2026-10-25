# TT_GREB_Reproduce
复现 TT-GREB 论文实验
# 项目简介
本仓库复现 TT-GREB 论文实验，包含 5 个图数据集（Cora、Amazon-Photo、OGB-Arxiv、Twitch-E、Elliptic）的加载与校验脚本
# 环境要求
- Python 3.9+
- PyTorch 2.6
- PyTorch Geometric 2.6.1
- ogb、numpy
# 安装
pip install torch torch_geometric ogb numpy
# 目录结构
dataset/
    ├── Amazon/
    │   └── Photo/
    │       ├── raw/amazon_electronics_photo.npz
    │       └── processed/          # 首次运行自动生成
    ├── Cora/
    │   └── Cora/
    │       ├── raw/                # ind.cora.* 共 8 个文件
    │       └── processed/          # 首次运行自动生成
    ├── OGB/
    │   └── ogbn_arxiv/
    │       ├── raw/                # arxiv.zip 等
    │       ├── mapping/
    │       ├── split/time/         # train/valid/test.csv.gz
    │       └── processed/          # 首次运行自动生成
    ├── Twitch/
    │   └── raw/
    │       ├── musae_DE_features.json
    │       ├── musae_DE_edges.csv
    │       └── musae_DE_target.csv
    └── Elliptic/
        └── raw/
            ├── elliptic_txs_features.csv
            ├── elliptic_txs_edgelist.csv
            └── elliptic_txs_classes.csv
# Cora
1.组织结构 ：Planetoid 自动加载，dataset[0] 返回整张 Data 对象，raw 为 ind.cora.* 共 8 个文件
2.是否划分 ：已划分，为掩码方式 
训练/验证/测试的区别体现在三个布尔掩码上（互不重叠）：
训练集节点数为140，掩码属性为`train_mask`
验证集节点数为500，掩码属性为`val_mask`
测试集节点数为1000，掩码属性为`test_mask`
（三者之和为 1640，小于总节点数 2708。剩余 1068 个节点为未参与划分的无标签节点，属官方标准现象，非数据异常）
3.用法
data = dataset_cora[0]
x_train = data.x[data.train_mask]   # torch.Size([140, 1433])

# Amazon-Photo
1.组织结构 ：Amazon 自动加载，raw 为单个 .npz
2.是否划分 ：未划分
Data 对象属性仅有 ['y', 'edge_index', 'x']， 不含任何 mask 
3.划分参考：
data = dataset_amazon[0]
num_nodes = data.num_nodes   # 7650
perm = torch.randperm(num_nodes)
train_idx = perm[:int(0.6 * num_nodes)]
val_idx   = perm[int(0.6 * num_nodes):int(0.8 * num_nodes)]
test_idx  = perm[int(0.8 * num_nodes):]
（具体比例以 TT-GREB 论文设定为准）

# OGB-Arxiv
1.组织结构 ：PygNodePropPredDataset 加载，划分 单独存文件 ：split/time/下的 train/valid/test.csv.gz
2.是否划分 ：已划分，独立索引文件方式，通过 get_idx_split() 读取，返回 dict，key 为 train / valid / test：
数据集            节点数 
train           	90941
valid             29799
test              48603      
（三者之和 169343 = 总节点数， 全覆盖，无遗漏 ）
3.用法 ：
split_idx = dataset_ogb.get_idx_split()
x_train = data.x[split_idx['train']]


# 运行
运行主程序：python main.py

# 常见问题
PyTorch 2.6 加载缓存报错 ：需在任何 torch.load 之前调用torch.serialization.add_safe_globals([...]) 注册安全全局类
路径解析错误 ：不要混用绝对路径和裸相对路径，建议统一基于脚本位置拼出数据根目录










