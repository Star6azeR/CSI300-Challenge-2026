# FNR 支持代码

**FNR（Factor Neighborhood Ranker，因子邻域排序模型）**结合冻结的截面因子表示、股票邻域信息、历史窗口统计与受约束重排。

## 目录与入口

```text
submission/
├── inference.py             # 平台推理入口：InferenceModel
├── train.py                 # 完整离线训练入口
├── requirements.txt         # CPU 推理依赖
├── requirements-training.txt
├── fnr/
│   ├── inference/           # 预测流程、冻结组件加载、NumPy 数值推理及排名合并
│   ├── features/            # 日频特征、有限窗口计算与市场状态
│   ├── models/              # 因子编码器、邻域修正头与统计修正
│   ├── training/            # 数据准备、分阶段训练与权重导出
│   ├── configs/             # 固定训练配方、排序策略及精度约定
│   └── paths.py             # 包内配置路径解析
└── model/                   # 已训练的冻结权重、预处理参数与校验清单
```

推理主流程见 [`fnr/inference/predictor.py`](fnr/inference/predictor.py)，离线训练调度见 [`fnr/training/pipeline.py`](fnr/training/pipeline.py)。包内模块使用相对导入，配置路径相对于包位置解析；数据和输出位置通过命令行传入。

`features/daily.py`用于离线训练，`features/bounded.py`用于推理时批量计算指定截面，两者保留各自的计算顺序。`model/`保留原提交的全部48个文件及其内部目录，权重与原冻结元数据逐字节一致。write-up和实验记录中的历史名称为`RH50_H60_K3_T240_FINAL`。

## 推理

以下命令在本发布目录根目录（`submission/`的上一级）运行。环境为Python 3.11、CPU。

```sh
python -m pip install -r submission/requirements.txt
```

```python
from submission.inference import InferenceModel

model = InferenceModel("submission/model")
scores = model.predict(as_of_date, data)
```

`data`为包含`MarketData`、`Factor33_winsor`和`Barra_factor`三张原始长表的字典。三表以`日期`和`代码`为键，保留比赛提供的字段名与数据类型。`as_of_date`采用`YYYY-MM-DD`格式。每次调用需要截至当日的120个交易日；输出为当日完整Barra股票池的`date`、`code`、`score`，包含停牌股票。

模型先计算基础集成分数与两个邻域修正头，再构造已实现收益的窗口统计和表示修正。最终保留参考信号前60名及其顺序，重排其余股票。名单由当次输入计算，固定策略见[`model/ranking_policy.json`](model/ranking_policy.json)。比赛字段`is_st`按最终数据处理约定置零。

推理只需运行依赖，使用NumPy和LightGBM载入权重。数值计算采用单线程，调用结束后恢复原线程设置。所有权重在初始化时载入。

## 完整训练

```sh
python -m pip install -r submission/requirements-training.txt
python submission/train.py --data-dir ../public_data --model-dir artifacts/retrained
```

数据目录应含三张表完整的2016–2022年月度文件，布局为`表名/年/月/YYYYMM.parquet`。输出目录须为空；训练数据需自行合法取得。

训练流程依次构建基础集成、因子表示、双邻域修正头、统计先验与最终修正头，并生成模型清单。固定配方位于[`fnr/configs/`](fnr/configs/)，随机种子为42。训练需要额外的PyTorch等依赖；完整重训的时间与内存需求高于推理。

也可通过模块方式查看各阶段参数，例如：

```sh
python -m submission.fnr.training.base --help
python -m submission.fnr.training.export --help
```

## 时间边界与数值约定

训练标签为H=1的后复权开盘收益`O(s+2)/O(s+1)-1`；冻结参数使用的训练标签退出日期早于2023-01-01。

推理统计只使用当次120日窗口内已完整实现的历史标签，最新样本为`s=t-2`。统计窗口为20日、半衰期为20日；先验、正则、特征选择与限幅均固定。临时回归系数在本次计算中使用，跨调用不保存。历史神经表示按统计所需截面重建，组合计算依赖保持在输入窗口内。

平均同分排名、浮点类型转换、稳定排序与舍入顺序均保留。冻结文件中的旧模块名或来源校验值记录原训练产物的来源，运行时通过当前包内代码加载权重。
