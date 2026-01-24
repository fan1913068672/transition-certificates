@echo off
echo Compiling main.tex...

REM First pass - compile tex to generate aux files
pdflatex -interaction=nonstopmode main.tex

REM Run bibtex to process bibliography
bibtex main

REM Second pass - resolve references
pdflatex -interaction=nonstopmode main.tex

REM Third pass - finalize all references
pdflatex -interaction=nonstopmode main.tex

echo.
echo Compilation complete!
echo Output: main.pdf
echo.
echo Cleaning up auxiliary files...

REM Delete auxiliary files
del main.aux 2>nul
del main.log 2>nul
del main.bbl 2>nul
del main.blg 2>nul
del main.out 2>nul
del main.toc 2>nul
del main.synctex.gz 2>nul
del main.fdb_latexmk 2>nul
del main.fls 2>nul
del main.nav 2>nul
del main.snm 2>nul
del main.vrb 2>nul

echo Cleanup complete!
start main.pdf
pause
