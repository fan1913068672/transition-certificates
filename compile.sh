#!/bin/bash

echo "Compiling main.tex..."

# First pass - compile tex to generate aux files
pdflatex -interaction=nonstopmode main.tex

# Run bibtex to process bibliography
bibtex main

# Second pass - resolve references
pdflatex -interaction=nonstopmode main.tex

# Third pass - finalize all references
pdflatex -interaction=nonstopmode main.tex

echo ""
echo "Compilation complete!"
echo "Output: main.pdf"
echo ""
echo "Cleaning up auxiliary files..."

# Delete auxiliary files
rm -f main.aux main.log main.bbl main.blg main.out main.toc \
      main.synctex.gz main.fdb_latexmk main.fls main.nav \
      main.snm main.vrb

echo "Cleanup complete!"
