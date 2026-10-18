# Manuscript sources

1. From the repository root, generate every table, figure and number quoted in the text:
   `python ptopofl_v2.py --exp all --seeds 10 --out results_v2`
2. Compile: `cd paper && latexmk -pdf ptopofl.tex`.
   The manuscript reads `../results_v2/` (or `./results_v2/` if you copy it next to the `.tex` files, e.g. on Overleaf).

Any number that has not been generated yet is printed as a red `??` (see `fallback_macros.tex`).
The TMLR style files (`tmlr.sty`, `tmlr.bst`, `fancyhdr.sty`) are included.
