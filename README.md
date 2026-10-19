<p align="center" style="color: #D4A017;"><strong>⚠️ 本仓库内容仅供学习研究用 ⚠️</strong></p>
<p align="center" style="color: #D32F2F;"><strong>🚫 禁止以任何方式应用于实盘 🚫</strong></p>

# Factor Neighborhood Ranker

结合截面因子表示、股票邻域信息、历史窗口统计与受约束重排的股票截面预测模型。

本模型于 **[上海科技大学沪深300指数增强与截面选股数据科学挑战赛](https://datatech.geekpie.club/c/csi300-2026)** Private Leaderboard 位列 🏆**第 1** · **RankIC = 0.0825** · **Information Ratio = 5.2132**。

## 资源

| 内容 | 链接与说明 |
| --- | --- |
| 模型代码 | [submission/](submission/)：包含推理入口、训练代码、固定配置与冻结权重；详见[代码说明](submission/README.md) |
| 模型 write-up | [FNR_model_writeup.pdf](FNR_model_writeup.pdf)：模型结构、训练方法与实证结果 |
| 搜索与训练 write-up | [FNR_search_writeup.pdf](FNR_search_writeup.pdf)：搜索策略、时间验证与泛化分析；附[源文件及实验记录](writeups_aux/search_methodology/) |
| Agent skill | [constraint-driven-model-research](skills/constraint-driven-model-research/SKILL.md)：规则核对、基线复现、自适应搜索与模型交付流程 |

## 代码使用流程

使用 Python 3.11，以下命令在本目录执行。数据格式及完整接口见[代码说明](submission/README.md)；随包权重可直接用于推理。

### 1. 训练（可选）

自行准备2016–2022年的 `MarketData`、`Factor33_winsor`、`Barra_factor` 原始数据，指定数据目录与空的模型输出目录：

```sh
python -m pip install -r submission/requirements-training.txt
python submission/train.py --data-dir ../public_data --model-dir artifacts/retrained
```

### 2. 推理

安装推理依赖，加载随包权重；使用重训结果时，将模型目录改为 `artifacts/retrained`。

```sh
python -m pip install -r submission/requirements.txt
```

```python
from submission.inference import InferenceModel

model = InferenceModel("submission/model")
scores = model.predict(as_of_date, data)
```

`as_of_date` 为信号日，`data` 为截至该日120个交易日的上述三张表；输出包含 `date`、`code`、`score`。

### 3. 回测

使用 [skd-backtest](https://github.com/HenryZ16/skd-backtest)，将 `model.predict` 传入 `BacktestEngine` 的 `inference` 参数。在固定日期、费用及组合约束下评价 RankIC 和 Information Ratio，配置见两份 write-up 及实验附件。

随包权重使用2016–2022年数据训练；本地年度验证须按对应年度的训练边界分别复现。

## 评测结果与研究材料

| Leaderboard | RankIC | Information Ratio | Ranking |
| --- | ---: | ---: | ---: |
| public | 0.0850 | 5.5684 | 2 |
| private | 0.0825 | 5.2132 | 1 |

上述成绩由比赛平台给出。本地训练方法、年度验证和模型选择过程见两份 write-up；LaTeX 源文件、搜索记录与提交成绩位于 [writeups_aux/](writeups_aux/)。材料索引、图表重建与编译命令见[附件说明](writeups_aux/search_methodology/README.md)。

## Agent skill

[约束驱动竞赛建模](skills/constraint-driven-model-research/SKILL.md)将研究中的操作经验整理为可供 agent 使用的通用流程，覆盖规则与数据审计、验证设计、官方基线复现、小批实验、动态剪枝及模型确认。具体指标、数据边界和资源预算由当前比赛确定。

在 Codex 中，将完整的 `skills/constraint-driven-model-research/` 目录复制到 `~/.codex/skills/`，在新会话中调用：

```text
$constraint-driven-model-research
根据当前比赛的规则、数据与官方基线，建立验证方案，在预算内实际训练并动态调整搜索，保留完整实验记录。
```

其他 agent 可直接读取 `SKILL.md`，并按其中链接加载参考文件。使用该流程无需加载本仓库的模型权重。

## 许可

Apache-2.0
