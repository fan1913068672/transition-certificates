@echo off
setlocal

echo Compiling Elsevier CAS manuscript (main.tex)...
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

echo Done. Output: main.pdf
endlocal
