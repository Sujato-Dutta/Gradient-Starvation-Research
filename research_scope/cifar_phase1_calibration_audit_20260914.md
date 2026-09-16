# CIFAR Phase-1 calibration audit — 14 September 2026

Status: **exploratory calibration only; no B run, no frozen preregistration, and no Gradient Starvation claim.**

## Evidence and integrity

The four DGX pilot directories are preserved in the ignored local archive
`results/cifar_phase1_pilots_20260914_4runs.tar.gz`, SHA-256
`62017064a23a1585cc1bafbeaf24f9cadd3d4c9df30d8717575e7b8da94637c5`.
The transfer hash matches the DGX-side hash. All four runs contain readable
`metadata.json`, a nonempty `final_model.pt`, and a finite 31-point trace from
update 0 through 4,710 in increments of 157. The final trace values match
the metadata. This audit read the archive without extracting its checkpoint
files or loading model pickles.

Content-addressed, locally verified pilot manifests are
`research_scope/cifar_weak_only_calibration_manifest.json` (SHA-256
`c119544e2ba514d44d24ea58fe4d9523bbc815cb451aa71ad46700cca812f525`)
and `research_scope/cifar_cue_only_feasibility_manifest.json` (SHA-256
`ff0b3fe0968c7526eb5829ea27c68472e4fec091c885b9bb4b30dddd3926863e`).
Each records the individual metadata, trace, and checkpoint-file hashes. Their
contents were checked against the transferred archive and local calibration
source bytes. They are still uncommitted exploratory records, not a freeze.

All four metadata records report the same draft-config SHA-256
`c1daed9634dfc5b99eae968e22181f9fa1f5020197337a99b36be1407b6cf663`,
protocol-code SHA-256
`c70e903951498384db25fbd632e78a49e77198de3bcb67870abf5aad4b42c634`,
runner-code SHA-256
`b39be16eeb13d3ff5366045edae1ccc85330f02070f5fc605642185b1fca853a`,
and CIFAR-10 archive SHA-256
`6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce`.
The three local code/config hashes agree with the files currently in this
working tree. Reported environment: Python 3.10.12, PyTorch 2.5.1+cu121,
torchvision 0.20.1+cu121, NVIDIA A100 MIG 1g.5gb. All used SGD momentum 0.9,
learning rate 0.05, weight decay 0.0005, batch size 256, and a cosine schedule
to update 4,710. Within each seed, W and cue-only have identical split and
initial-weight hashes; the two seeds have different hashes.

## Results

Validation has 500 images per class, disjoint from the 4,000-per-class
training split. The original CIFAR test set has not been used here.

| Condition | Seed | Cue amplitude | Final validation accuracy | Final balanced Brier skill | Cross-entropy | Logged training time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Weak-only, neutral ribbon | 13001 | 0 | 84.28% | 0.7557 | 0.4476 | 1,237 s |
| Weak-only, neutral ribbon | 13002 | 0 | 83.96% | 0.7484 | 0.4659 | 1,255 s |
| Cue-only, blank core | 13001 | 0.5 | 90.46% | 0.7980 | 0.5354 | 1,250 s |
| Cue-only, blank core | 13002 | 0.5 | 89.26% | 0.7745 | 0.5897 | 1,231 s |

Both W pilots pass the draft's 80% neutral-accuracy and 0.25 Brier-scale
minimums. Cue-only performance is near the 90% cue reliability, so amplitude
0.5 is feasible without using any B outcome to select it. The observed 90.46%
in one finite validation split is not evidence of exceeding the population
noise ceiling.

For the W traces, the first logged Brier-skill hit of 0.5 is at update 1,099
(seed 13001) and 942 (seed 13002); the first hit of 0.6 is at 1,256 and
1,727. The first logged 80%-accuracy hits are at 2,512 and 2,826. These are
**checkpoint-grid observations**, not continuous-time hitting times. The W
terminal Brier-skill mean is approximately 0.7520.

## Prospective choices and remaining gates

Based **only** on these W and cue-only pilots, a defensible candidate is
`H=4710` updates, checkpoint spacing 157 updates, cue amplitude 0.5,
`beta=0.5` Brier skill, and a rounded fixed response scale `S=0.75`.
`beta=0.5` gives both W seeds a nondegenerate hit before one quarter of H and
leaves a substantial margin below their terminal responses. These values
are **recommendations, not frozen settings**. In particular, these pilots
measure Brier skill on the validation split, whereas the planned confirmation
response uses a separate fixed neutral diagnostic split. The diagnostic
response, gate, and materiality calculations must be implemented and tested
before a frozen protocol can claim those definitions.

Before any B training: the locally staged paired B/W trainer, same-input and
exact-ablation checks, neutral/random/conflict behavioral evaluation,
weak-response trajectory and first-hit/deficit analysis, E4-lite head test,
failure handling, and output/schema audit require further review and DGX
compatibility checks. Review and commit the content-addressed W and cue-only
calibration manifests, pin a reviewed source commit and complete environment manifest,
reconcile the paper/E0 claims, and freeze the preregistration with all
checker-required fields. Confirmation seeds 33001–33008 remain untouched.
The current calibration runner explicitly refuses B training, and the new
paired CLI explicitly refuses a draft preregistration.

## Locally staged paired implementation (not run on DGX)

`src/gradient_starvation/cifar_matched.py` produces B and W from a single
sampled/augmented image, with W made by exact zeroing of B's appended ribbon.
It matches the standalone W DataLoader shuffle path; a tiny fake-CIFAR test
checks identical final W model weights. The same neutral diagnostic IDs are
used for both checkpoint response trajectories. Signed and positive-part
deficit AUC and material-deficit duration use a stated linear interpolation
on the logged update grid; they do not claim continuous-time paths. Final
behavior uses common neutral, uniform random, consistent, and nine wrong-code
conflict views per source image.

`src/gradient_starvation/cifar_probe.py` implements E4-lite with frozen
encoders, a disjoint neutral probe-fit split, fit-only feature scaling,
zero-initialized balanced linear-softmax heads, a common L2 grid, and a fixed
LBFGS iteration budget. The selected head's fit-gradient norm is reported.
`scripts/run_cifar_matched.py` has no draft/debug override: it requires a
committed frozen protocol, matching reviewed source bytes, environment and
dataset hashes, and an active `gpu_student` allocation. It writes incremental
trace/checkpoint files and marks interrupted blocks incomplete.
`scripts/aggregate_cifar_matched.py` verifies all eight blocks, including
trace/checkpoint hashes, before reporting paired 95% t intervals; missing
blocks yield no confirmatory interval. It now carries the B arm, W arm, and
B-minus-W seed-block summaries for neutral, random, consistent, and conflict
accuracy into the aggregate rather than leaving random-cue results only in
per-run JSON. `scripts/record_cifar_environment.py` records an environment-v2
runtime fingerprint including the exact package set, platform, CUDA/cuDNN,
GPU memory, compute capability, and multiprocessor count; confirmation checks
these fields live. `scripts/freeze_cifar_preregistration.py` derives the frozen
protocol only from the W/cue-only manifests, reviewed source commit, and DGX
environment manifest. Peak CUDA memory is recorded by each paired run. These
source-state safeguards still require the DGX environment capture, reviewed
commit, and a DGX compatibility check before any real B training. Focused
local checks are:

```bash
MPLBACKEND=Agg python -m pytest -q tests/test_cifar_freeze.py tests/test_cifar_matched.py tests/test_cifar_aggregate.py tests/test_cifar_protocol.py tests/test_cifar_preregistration_draft.py
```
