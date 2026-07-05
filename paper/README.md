# `paper/` — manuscript hub

This directory is a **shared-asset hub**. The evidence (tables and figures) is
generated once from the analysis pipeline and shared by every manuscript variant,
so results never diverge across variants.

## Layout

```
paper/
  tables/          Generated LaTeX tables (source of truth)
  figures/         Generated figures (source of truth)
  references.bib   Single shared bibliography
  arxiv/           The public manuscript
    manuscript.tex \input{../tables/..}, \includegraphics{../figures/..},
                   \bibliography{../references}
  scripts/         Tooling (Overleaf sync, etc.)
```

## Regenerating assets

Tables and figures are produced by the analysis pipeline — do not hand-edit them:

```bash
bash analysis/results/generate-paper-assets.sh          # tables + figures
bash analysis/results/generate-paper-assets.sh --tables # tables only
bash analysis/results/generate-paper-assets.sh --figures
```

The generators write directly into `paper/tables/` and `paper/figures/`.

## Building the manuscript

```bash
cd paper/arxiv && latexmk -pdf manuscript.tex
```

Any manuscript variant lives in its own folder and pulls the shared assets via
relative paths (`../tables`, `../figures`, `../references`), so a variant can add
a local override simply by shadowing a file in its own directory.

## Overleaf sync

`scripts/push-overleaf.sh` performs a one-way push of the `arxiv/` manuscript plus
the shared assets to Overleaf. The Overleaf project's main document is
`arxiv/manuscript.tex`.
