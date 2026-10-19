# FNR 模型 write-up 源文件


[write-up PDF](../../FNR_model_writeup.pdf) · [LaTeX源文件](FNR_model_writeup.tex)

在本目录用XeLaTeX编译：

```sh
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build FNR_model_writeup.tex
cp build/FNR_model_writeup.pdf ../../FNR_model_writeup.pdf
```

图示内嵌于TeX，编译需要ctex、TikZ及常见数学宏包。
