# Frozen CIFAR matched-confirmation audit — 17 September 2026

Status: **confirmation succeeded; independent compact-evidence audit passed**.

This report audits the preregistered one-architecture CIFAR-10 experiment at source commit `0726aea12947a30eb60ae32fc9c1eb0a7a122660`. Eight fresh paired seed blocks compare **B = core + removable label-correlated ribbon cue** with **W = the identical core and a neutral zero cue channel**. Pair members share initialization, sample order, augmentation randomness, optimizer settings, splits, and training horizon. The primary endpoint was frozen before B outcomes were inspected.

## Evidence and integrity

- Compact archive: `results/cifar_confirmation_compact_dde8d35.tar.gz`
- Archive SHA-256: `4310e61a825b9be939063e59ab27f229a3b4545c819b2de2c610ef1620e7bea8` (matches the DGX source)
- Frozen config SHA-256: `8f662f9982f8445bc144a021b9717b2986ee7d4f7d19c03c9a1868f5aa1e2fea`
- Environment SHA-256: `1c1e2d126093f449e176a9ef22e637a8a52ceb0ac0574028f7930e0e2266a034`
- CIFAR-10 archive SHA-256: `6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce`
- Slurm array `23248`: all eight tasks completed with exit code `0:0`; all eight stderr logs are empty.
- Every run has 31 expected checkpoints through update 4,710, matching trace/summary/manifest hashes, byte-identical frozen config and environment snapshots, and identical B/W weights at initialization.
- The eight seeds have eight distinct initialization, split, and training-order hashes. All 240 post-initialization checkpoint gaps (30 per seed) are negative.

The compact archive deliberately excludes `.pt` payloads. It contains their manifests and hashes, but cannot by itself re-hash the model files. Keep the full checkpoint directories on DGX or another durable store until review/rebuttal is over.

The reproducible independent audit is `scripts/audit_cifar_confirmation_results.py`; its machine-readable output is `research_scope/cifar_confirmation_audit_metrics_20260917.json`. It recomputes the results from the eight raw summaries and traces rather than trusting the aggregate file.

## Confirmatory result

All **8/8** W runs passed the learnability gate, and all **8/8** paired blocks passed the preregistered primary causal-starvation certificate: at the first update where W reached the frozen response target, B's core response was lower.

| Endpoint | Mean over paired seeds | Two-sided 95% t interval | Frozen target |
|---|---:|---:|---:|
| Signed normalized deficit AUC | 0.489 | [0.413, 0.564] | at least 0.10 |
| Material-deficit duration / horizon | 98.3% | [97.6%, 99.0%] | at least 10% |
| B−W Brier-skill gap at W first hit | −0.399 | [−0.490, −0.307] | negative |
| Final B−W Brier-skill gap | −0.315 | [−0.361, −0.269] | descriptive |

The signed and positive-part AUCs are equal in every seed, so there is no compensating positive-gap area on the frozen checkpoint grid. The first W gate hits occur between updates 785 and 1,256. This is a large, persistent, seed-consistent effect rather than a single-run or late-checkpoint anomaly.

## Behavioral meaning

Final-checkpoint mean accuracies on the fixed held-out diagnostic data were:

| Evaluation | B: core + cue | W: core only | Paired B−W gap (95% CI) |
|---|---:|---:|---:|
| Neutral/core | 64.0% | 83.9% | −19.9 pp [−22.6, −17.2] |
| Random cue | 54.7% | 82.3% | −27.6 pp [−28.7, −26.4] |
| Label-consistent cue | 99.1% | 82.2% | +16.8 pp [+15.8, +17.9] |
| Conflict cue | 50.5% | 82.4% | −31.9 pp [−33.0, −30.8] |

The cue therefore makes the B model nearly perfect when it agrees with the label, but substantially worse when the cue is absent, random, or wrong. Because W learns the core well and the pair differs only through the removable cue intervention, this is strong controlled evidence that cue availability causally suppresses useful core learning in this cell.

## E4-lite interpretation

With the original trained heads, the neutral B−W gap is **−19.9 pp** [−22.6, −17.2]. With equal fresh balanced linear heads fitted to frozen neutral features, the gap shrinks to **−10.8 pp** [−11.1, −10.4]. The mean recovery is **+9.1 pp** [+6.5, +11.8], but every seed retains a negative fresh-head gap.

The correct conclusion is mixed: **some harm is readout underuse, and a substantial remainder is reduced linear accessibility under this probe**. This does not establish information destruction or failure of every nonlinear decoder.

## Relation to the theorems

This experiment directly instantiates the paper's matched, learnability-gated **causal certificate** in a controlled CNN image setting. It strongly supports the practical relevance of the finite-width/special-regime theory.

It does **not** experimentally verify the exact continuous-time cross-kernel response-flow identity or the tail-area iff theorem. The CIFAR run uses discrete minibatch SGD and checkpointed multiclass Brier skill; it does not measure the theorem's instantaneous full-batch logistic-gradient-flow drift. The paper must present this as a controlled empirical realization of the certificate, not as a numerical proof of the tail-area theorem or a universal law for arbitrary networks.

## ICLR assessment and remaining work

This result is strong enough to serve as the paper's flagship empirical demonstration. It materially upgrades the work from synthetic/special-regime evidence to a preregistered, reproducible controlled-image result. It is **not, by itself, a guarantee of ICLR acceptance or an oral**: one artificial cue and one architecture establish existence and diagnostic usefulness, not prevalence or universality.

Before submission, the critical path is:

1. Correct the E0 manuscript counts and labels already identified in `research_scope/e0_consistency_audit_20260914.md`; regenerate affected tables/figures from the audited run-level source.
2. Integrate this CIFAR result and the E4 wording into the nine-page main paper, with all eight seed points and confidence intervals in the appendix. Keep theory and empirical claims explicitly separated.
3. Preserve the full DGX checkpoints and archive their hashes. Run one final held-out CIFAR test-set evaluation only if the fixed protocol/model-selection rules make it a clean one-shot check; label it post-preregistered if it was not frozen.
4. Obtain independent proof review and either reproduce the external parameter-certificate package or remove claims that imply it was independently verified.
5. Run the full test/readiness suite, rebuild both PDFs, check anonymity/page limits/AI-use statement, and audit every retained number against its pinned artifact.

E5-lite is optional. It is useful mechanism evidence, but less important than correcting E0 and integrating the successful confirmatory result without overclaiming.
