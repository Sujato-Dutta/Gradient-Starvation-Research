# ICLR 2027 manuscript build

This directory is an auditable anonymous ICLR wrapper around the canonical scientific source in `paper/main.tex`. It is a reconstruction, not a claim of byte-identical recovery of the older 13-page PDF: the older TeX/Bib dependency tree is unavailable.

## Build

From `paper/`:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=iclr2027 iclr2027/submission.tex
```

The output is `paper/iclr2027/submission.pdf`. The generic canonical rendering remains:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

`submission.tex` defines `\ICLRSubmission` and inputs `paper/main.tex`; no separate
manuscript fork is maintained. In ICLR mode, the same file emits a compact
reviewer-facing synthesis as an additional claim surface, followed by the
required/recommended disclosure statements and references, then the complete
theorem, experiment, result, audit, and proof package as appendices. The compact
and detailed surfaces are explicitly cross-checked in `review_resolution.md`;
claim-bearing wording must remain synchronized.

## Format and scope

- `iclr2027_conference.sty` is a verbatim copy of the official ICLR 2027 style file. SHA-256: `797deef41724e93761426ac0cbcca46279a91cc650dd1f0ce76a4f08d2098ea6`.
- The consistent author-year bibliography uses the standard `plainnat` style.
- `\iclrfinalcopy` remains disabled: the PDF is anonymous and carries the under-review banner and line numbering.
- The main-text end page is recorded by `page:main-end` in `submission.aux`; references, disclosures, and appendices are outside the scientific main-text count as specified by the venue guidance.
- No real-data result is claimed. The Waterbirds adapter remains unrun and outside the theorem's matched full-batch setting.
- Rebuilding the terminal beta profile is a read-only derivation from archived E-NL outcomes; it does not rerun training.

See `review_resolution.md` for the itemized Claude-review resolution tape and validation record.
