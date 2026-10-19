# 面向股票截面预测的约束驱动模型搜索与训练方法


[write-up](../../FNR_search_writeup.pdf) · [LaTeX源文件](search_methodology.tex)

## 目录

- `evidence/`：按研究主题整理的报告、实验记录与归档源码；`index.json`保存包内路径、原文件路径及两份校验值。
- `data/submission_curve.csv`：33条提交成绩，仅含原始文件编号、文件名、RankIC与IR。
- `tools/`：绘图与校验脚本，均只依赖Python标准库。
- `third_party/`：第三方源码许可及来源。
- `MANIFEST.json`：分发文件校验清单。

## 使用

在本目录执行：

```sh
python tools/materials.py verify
python tools/redraw_figure.py
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build search_methodology.tex
```

编译需要XeLaTeX、ctex、TikZ及常见数学宏包；字体优先宋体、黑体，否则使用Fandol。
将重新编译的PDF更新到发布包根目录：

```sh
cp build/search_methodology.pdf ../../FNR_search_writeup.pdf
```

图形均内嵌于TeX。归档源码用于核对方法和参数，完整训练还需上游模块与合法取得的数据；历史报告中的产物名保留当时的运行语义。
发布范围与许可状态见[发布说明](RELEASE_NOTES.md)。
