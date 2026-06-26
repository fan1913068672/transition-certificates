@echo off
REM A simple batch script to compile the LaTeX cover letter.

REM Check if pdflatex is installed
where pdflatex >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: pdflatex command not found.
    echo Please make sure you have a LaTeX distribution installed.
    exit /b 1
)

REM Compile the LaTeX file
echo Compiling letter.tex...
pdflatex letter.tex

REM Run again to ensure cross-references are correct
pdflatex letter.tex

echo Compilation finished. Output is in letter.pdf

REM Clean up auxiliary files
if exist *.aux del *.aux
if exist *.log del *.log
if exist *.out del *.out

echo Auxiliary files cleaned up.
echo.
pause
