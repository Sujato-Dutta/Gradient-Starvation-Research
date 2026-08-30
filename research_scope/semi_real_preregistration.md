# Semi-real generated-cue causal study

## Status

**Scientific design frozen; outcome generation blocked.** The machine-readable
contract is `configs/semi_real_preregistration.yaml`, validated by
`scripts/check_semi_real_preregistration.py`. It currently and intentionally reports
`execution_authorized: false`.

No MNIST, FashionMNIST, torchvision environment, selected-index manifest, optimizer,
training trajectory, or result was created during this work. This document contains
no empirical claim.

## Why this replaces Waterbirds as the causal validation design

The current Waterbirds adapter cannot remove the background while preserving the
bird exactly, and its second projected coordinate is an intercept rather than an
identified bird/core response. It remains an unrun surrogate and cannot answer the
requested causal question.

The semi-real design instead uses two real-image sources:

- MNIST, classes 3 versus 8;
- FashionMNIST, classes 0 versus 6.

The real pixels are the weak/core feature. A generated, fully label-aligned patch is
placed in a separate model input channel. The BOTH and WEAK tensors share the exact
same labels, stable source IDs, and real-pixel channel. WEAK is formed only by
zeroing the generated channel. This is a known intervention rather than an inferred
background edit.

It is correctly called **semi-real**: the core images are real benchmark data, but
the dominant cue and its intervention are generated.

## Frozen preprocessing and intervention

Raw official uint8 pixels are mapped by `x/127.5-1`, resized deterministically to
14 by 14 with area interpolation, and never augmented. The generated cue is a 2 by
2 upper-left patch of amplitude `4 y_signed` in channel 1. Channel 0 contains only
the real image. The weak-only member has channel 1 identically zero.

For every record, evaluation must fail unless all of the following are exact:

1. labels match;
2. stable source sample IDs match;
3. the core channel is bitwise equal;
4. the weak generated channel is all zero;
5. the generated channel is the only difference.

`semi_real.make_paired_semi_real_task` implements this operation. Its tests use toy
uint8 images and generate no scientific outcome.

## Identified common response functionals

A fixed, disjoint, cue-free real-image probe defines both functionals for both
causal conditions:

```text
M_w(theta) = mean_i y_i f_theta(core_i, cue=0),
M_s(theta) = (1/2) mean_i y_i [
                 f_theta(core_i, cue=+rho y_i)
               - f_theta(core_i, cue=-rho y_i)].
```

`M_w` is therefore an identified response to real core content, not an intercept.
`M_s` is the aligned-versus-conflicted response to the known generated cue on the
same real cores. Both are functions of model parameters and the frozen probe only;
they do not depend on whether the model was trained in BOTH or WEAK. Shared
initialization therefore gives an exactly zero initial paired response gap.

The primary derivative is always the universal direct-autograd quantity

```text
d M_a/dtau = -<grad M_a, grad L_condition>.
```

`semi_real.semi_real_statistics` evaluates it without populating `.grad`. The
nonlinear image logits are not assumed to lie in the span of `(M_s,M_w)`, so the
synthetic two-probe `Gg` projection is explicitly non-promotable.

Responses are reported as gains from the shared initial value. This makes the weak
target `beta=0.5` a learning increment rather than a random absolute initialization
threshold; subtracting the constant does not change its derivative.

## Model and paired optimization

The frozen model is a small two-channel CNN with convolution widths 16 and 32 and
no pretraining. This deliberately moves the validation beyond synthetic recurrent
inputs while retaining the general finite-width response-flow theorem. The future
paired evaluator must use independent full-batch SGD trajectories from one exact
initial state, with learning rate 0.01, 500 steps, no momentum, weight decay,
clipping, or augmentation, and logging every five steps.

The design crosses four untouched data-selection seeds with eight untouched model
seeds for each dataset, giving 64 dataset/data/model records. Reuse-aware inference
must resample both seed axes rather than treating all pair rows as independent.

## Frozen causal endpoints

Primary endpoints are:

1. the weak/core response AUC gap;
2. the learnability-gated causal-starvation certificate;
3. weak-only learnability.

The certificate retains the existing hierarchy:

- an exact drift crossing is rate suppression only;
- a response gap below zero is outcome suppression;
- causal starvation additionally requires a finite weak-only first hit;
- multiple drift sign changes and the finite-step tail-area result are always
  reported rather than collapsed into a single clean crossing.

Secondary sanity checks include core-only, BOTH, and cue-conflict accuracy, GSI-5,
response/drift crossings, and crossing times. Accuracy is not a substitute for the
response functionals.

The preregistered acceptance rule requires complete records and exact intervention
checks, at least 75% weak-only learnability and 50% causal certificates separately
on each dataset, and a strictly positive lower endpoint for the reuse-aware 95%
bootstrap interval of the weak AUC gap. These thresholds may fail; they cannot be
changed after outcomes to rescue the hypothesis.

## Authorization gates

The design is not an execution-ready preregistration. All of the following remain
mandatory:

1. reviewed clean Git source;
2. an exact Torch/torchvision environment lock and digest;
3. locally present official raw files and their digests;
4. a content-addressed manifest of disjoint train/probe/evaluation indices;
5. untouched-seed attestation;
6. a committed preregistration revision before outcome generation;
7. `submission_claim_set.yaml` changed through review to allow new outcomes.

The working tree is dirty, no vision data or optional dependencies are present, and
no commit was authorized. Therefore neither dataset download nor training was
attempted. Running

```text
.venv-test/bin/python scripts/check_semi_real_preregistration.py
```

validates the design while returning `execution_authorized: false`. Adding
`--require-authorized` fails by construction. Any eventual transition to an
authorized schema must be separately reviewed; editing null digests in this blocked
artifact is not authorization.
