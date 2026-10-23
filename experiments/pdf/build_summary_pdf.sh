#!/usr/bin/env bash
# Build results/SUMMARY.pdf from results/SUMMARY.md (pandoc 3 + xelatex).
# template.tex = `pandoc -D latex` without \usepackage{lmodern} (not installed here).
set -euo pipefail
cd "$(dirname "$0")/../../results"
F=/usr/share/fonts/truetype/liberation/
pandoc SUMMARY.md -f markdown-implicit_figures -o SUMMARY.pdf --pdf-engine=xelatex \
  --template=../experiments/pdf/template.tex --resource-path=. \
  -V geometry:margin=1.8cm -V fontsize=10pt -V colorlinks=true \
  -V mainfont=LiberationSans \
  -V mainfontoptions="Path=$F,Extension=.ttf,UprightFont=*-Regular,BoldFont=*-Bold,ItalicFont=*-Italic,BoldItalicFont=*-BoldItalic" \
  -V monofont=LiberationMono \
  -V monofontoptions="Path=$F,Extension=.ttf,UprightFont=*-Regular,BoldFont=*-Bold,ItalicFont=*-Italic,BoldItalicFont=*-BoldItalic"
echo "wrote $(pwd)/SUMMARY.pdf"
