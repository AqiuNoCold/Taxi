# 东南大学AI作业 - NYC出租车行程时长预测

## 任务描述

**比赛链接**: https://www.kaggle.com/competitions/nyc-taxi-trip-duration/overview

**目标描述**: 根据数据集中的信息(经纬度,开始时间,乘客数量,vendor_id,store_and_fwd_flag),预测出租车所需耗时(目标),训练模型.最终将结果提交至网站上.

**提交链接**: https://www.kaggle.com/competitions/nyc-taxi-trip-duration/submissions

## 数据集特征

原始数据包含以下字段：
- `pickup_datetime`: 上车时间
- `dropoff,datetime`: 下车时间  
- `pickup_longitude/latitude`: 上车经纬度
- `dropoff_longitude/latitude`: 下车经纬度
- `passenger_count`: 乘客数量
- `vendor_id`, 供应商ID
- `store_and_fwd_flag`: 存储转发标志
- `trip_duration`: 行程时长（目标变量）

## 特征工程与数据预处理

### 核心特征提取方法

#### 1. 时间特征提取
```python
df_new["pickup_hour"] = df_new["pickup_datetime"].dt.hour
```
- 提取上车时间的小时信息
- 捕捉交通高峰期和非高峰期的差异

![alt text](imgs/pickup_hour_distribution.png)

#### 2. 地理特征工程
```python
df_new["distance_km"] = haversine(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat)
```
- 使用**Haversine公式**计算地球表面两点间的最短距离
- 考虑地球曲率的精确地理计算

![alt text](imgs/distance_distribution.png)

#### 3. 速度特征衍生
```python
df_new["speed_kmh"] = df_new["distance_km"] / (df_new["trip_duration"] / 3600)
df_new.loc[df_new["speed_kmh"] > 200, "speed_kmh"] = np.nan
```

![alt text](imgs/speed_distribution.png)

- 计算平均行驶速度
- 异常值检测与清洗（去除>200km/h的不合理速度）
- 去除离散经纬度地点


![alt text](imgs/dropoff_locations.png)

![alt text](imgs/pickup_locations.png)

#### 4. 行驶时间特征

![alt text](imgs/duration_distribution.png)

### 数据预处理流程
1. **时间格式转换**: datetime标准化处理
2. **地理距离计算**: Haversine公式应用
3. **异常值处理**: 速度阈值过滤
4. **缺失值处理**: 自动识别和处理

## 特征分析与建模

### 模型选择：随机森林回归器

**选择随机森林的核心原因**:

1. **混合特征处理能力**
   - 自动处理分类特征（vendor_id）和连续特征（经纬度）
   - 无需特征标准化和编码

2. **内置特征重要性评估**
   ```python
   model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
   importance = model.feature_importances_
   ```

3. **地理空间数据适应性**
   - 捕捉经纬度间的非线性空间关系
   - 自动识别交通热点区域

4. **时间模式学习**
   - 处理pickup_hour的周期性特征
   - 学习不同时段的交通规律

5. **鲁棒性与效率**
   - 对出租车数据中的异常值不敏感
   - 支持并行计算，处理大规模数据

## LightGBM训练

### 模型选择：LightGBM回归器

**与NYC出租车预测任务的适配性**：

1. **城市交通数据特性匹配**
   - 出租车行程时长受多因素非线性影响（距离、时间、地点、交通状况）
   - LightGBM擅长处理复杂的特征交互关系，能够自动学习不同区域、不同时段的交通模式

2. **大规模数据处理需求**
   - NYC数据集包含百万级别的行程记录
   - LightGBM基于直方图的优化算法，训练速度比传统GBDT快10倍以上
   - 内存效率高，适合处理大型城市交通数据

3. **回归预测场景优化**
   - 针对连续数值预测（行程时长）进行了专门优化
   - 内置对数变换支持，天然适配RMSLE评估指标
   - 相比随机森林，在回归任务上通常具有更高精度

### LightGBM核心技术特性

**梯度提升决策树优势**：
- **叶子增长策略**：采用叶子优先(leaf-wise)增长，比层次增长更高效
- **特征并行**：支持大规模特征的并行处理
- **网络通信优化**：分布式训练时通信开销更小

**内置正则化机制**：
- 自动特征选择和样本采样，防止过拟合
- 对噪声数据和异常值具有较强鲁棒性
- 适合处理出租车数据中的不规则行程记录

### 模型性能与结果

**训练效果**：
- 验证集RMSLE指标显著优于随机森林基线
Validation RMSLE: 0.4226616652012385

- 训练时间缩短至分钟级别，支持快速迭代优化
- 特征重要性分析显示距离和地理位置为关键预测因子

![alt text](imgs/re.png)

## 文件结构
```
├── data/
│   └── train.csv              # NYC出租车训练数据
├── utils.py                   # 核心工具函数库
│   ├── haversine()           # 地理距离计算
│   ├── preprocess_copy()     # 数据预处理管道
│   ├── visualize_all()       # 数据可视化套件
│   ├── feature_importance()  # 随机森林特征分析
│   └── plot_importance()     # 特征重要性可视化
├── feature.ipynb             # 特征工程分析notebook
├── test.ipynb               # 模型测试与验证
└── README.md                # 项目文档
```

## 分析工作流

1. **数据加载** → 读取NYC出租车历史数据
2. **特征工程** → 时间、地理、速度特征提取
3. **数据清洗** → 异常值检测与处理
4. **可视化探索** → 多维度数据分布分析
5. **模型训练** → 随机森林回归建模
6. **特征评估** → 重要性分析与可视化
7. **模型优化** → 基于特征贡献度调优

通过这套完整的机器学习管道，项目能够有效预测纽约市出租车的行程时长，为城市交通分析提供数据支持。

