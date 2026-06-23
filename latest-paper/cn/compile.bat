@echo off
setlocal
chcp 65001 > nul

echo Compiling Chinese LaTeX manuscript (main.tex) with XeLaTeX...
xelatex -interaction=nonstopmode main.tex
bibtex main
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex

echo Done. Output: main.pdf
endlocal
