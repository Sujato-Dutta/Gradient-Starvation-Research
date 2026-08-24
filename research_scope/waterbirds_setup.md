# Waterbirds: what the author must do before this can run

**Nothing in this file has been run.** The two CDC variants are implemented and
unit-tested against synthetic tensors, but no Waterbirds experiment has been
executed: no dataset was downloaded, no pretrained backbone was fetched, and no
result exists. Treat every statement below as a setup instruction, not a finding.

## Required setup

1. **Optional dependencies.** `pip install -r requirements-waterbirds.txt`, which
   supplies `torchvision` and `wilds`. The preflight test
   `test_waterbirds_dependency_preflight_is_actionable` asserts that a missing
   dependency produces an actionable error naming that file.
2. **Dataset.** WILDS will fetch Waterbirds when `data.download: true`. It is a few
   GB. Set `data.root` to somewhere with space. This is deliberately not automated
   here; downloading a dataset is the author's decision.
3. **Pretrained backbone.** `model.pretrained: true` pulls ImageNet weights through
   `torchvision`. Set `false` to train from scratch, which will change the
   conclusions substantially.
4. **Compute.** ResNet-18 on Waterbirds for 30 epochs across 5 methods and 3 seeds
   is a GPU job. `experiment.device: auto` will pick CUDA when present.

## The two CDC variants, and why the names matter

| method | group labels at training time | `information_setting` column |
|---|---|---|
| `erm` | no | `group_agnostic` |
| `spectral_decoupling` | no | `group_agnostic` |
| `counterfactual_drift_modal` | **no** | `group_agnostic` |
| `interaction` | yes | `oracle_uses_group_labels` |
| `counterfactual_drift_oracle` | **yes** | `oracle_uses_group_labels` |

`counterfactual_drift_oracle` builds its strong/weak coordinates from the
background annotation. It is a **mechanistic transfer probe**: it asks whether the
starvation mechanism identified on synthetic data is present on real data, given
oracle knowledge of the spurious cue. It is **not** comparable to group-agnostic
robustness methods and must never appear in the same comparison column as one.

`counterfactual_drift_modal` uses no group labels. It estimates the dominant mode
as the leading principal direction of the centred penultimate features, on the
hypothesis that the spurious cue is the largest source of representational
variance.

Every emitted row carries an `information_setting` column, and `run_waterbirds`
rejects any method outside the known set, so the distinction cannot be lost in
post-processing.

## The modal estimator's assumption is measurable, and must be measured

The modal variant is only the causal method if its leading component actually
tracks the background. `modal_estimator_agreement` returns the absolute correlation
between that component and the true background annotation, and the value is logged
per epoch as `modal_background_agreement`.

**Read it before trusting any modal result.** A low agreement means the rescue
direction is not the causal one, and the modal variant is then measuring something
else. This diagnostic uses group labels, so it is a *diagnostic only* — it must not
feed back into the method.

## Run the sensitivity pilot first

`configs/waterbirds_pilot.yaml` exists because the learning rate in
`configs/waterbirds.yaml` is now wrong. The CDC arms require plain SGD with zero
weight decay, and that requirement binds every arm so the comparison is not
confounded by the optimizer -- but `1e-4` was inherited from the AdamW setup, and
AdamW's per-parameter scaling makes it a very different step size from SGD's. Running
the full grid first risks a "CDC underperforms" result that is really a learning-rate
artifact. The pilot config lists the two sweeps and six acceptance criteria.

## Three honest limitations to state in any write-up

1. **There is no exact weak-only counterfactual on Waterbirds.** On synthetic data
   the target comes from a matched shadow model trained on exactly strong-ablated
   inputs. Waterbirds has no such ablation: you cannot remove the background and
   leave the bird unchanged. The implementation therefore uses a configured
   `mitigation.weak_drift_target` surrogate, which is a *stated design choice*, not
   the synthetic guarantee. Result 1 of `cdc_theorem.md` still holds relative to
   whatever target is supplied, because it is a property of the projection; but the
   target itself is no longer the exact counterfactual.
2. **The "weak coordinate" is an intercept, not an identified feature.** Both variants
   regress the margin on `(strong, 1)`, so the second coefficient is the average
   signed margin *unexplained* by the estimated strong mode. It is not a bird-shape
   response, and nothing in this implementation identifies one. Waterbirds is
   therefore a **surrogate** mechanistic probe, and stays one until that coordinate is
   validated against an independent core-feature readout. This applies to the oracle
   variant too: knowing the background does not identify the bird.
3. **The correction is applied to the classifier head only.** This matches the
   existing interaction penalty. Extending it to the backbone changes the protected
   geometry and would need its own validation, so it is not done implicitly.

## Suggested first run

Start small enough to detect wiring problems before spending GPU hours:

```bash
python run_experiment.py waterbirds --config configs/waterbirds.yaml \
  --set training.epochs=1 --set training.seeds='[0]' \
  --set mitigation.methods='[erm, counterfactual_drift_modal]'
```

Check that `summary.csv` carries `information_setting`, that
`modal_background_agreement` is present and plausible, and that
`cdc_strong_drift_change` is small for the CDC arms. Only then scale up.

## What a completed study must report

Average accuracy, worst-group accuracy, per-group accuracies with confidence
intervals, GSI-5, wall-clock and memory cost, and `modal_background_agreement`.
Oracle and group-agnostic arms must be tabulated separately. Per `a.md` §20, a real
data result does not license any claim about the unproved mean-field theory.
