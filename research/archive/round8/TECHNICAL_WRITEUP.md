# Selective feature-channel edits without training or external images

We tested whether targeting a small subset of feature channels could improve the sharpness-versus-grid tradeoff seen in round7. The experiment used the existing generator, a fixed set of analytical weight edits, and generated portraits as evaluation probes. There was no optimizer, backpropagation, conventional training, external image dataset, or pretrained restoration model.

**Result:** one candidate met all development criteria and all independent Gaussian-noise criteria. It did not meet the uniform-noise artifact criteria. We exported it as an experimental variant at 128×128 and 256×256, preserved the prior models, and did not promote it as a general replacement. Visual changes are subtle; there is no demonstrated recovery of missing high-resolution facial detail.

## Preserved parent

Parent checkpoint: `../round5/candidates/model_clean_diverse.pt`.

SHA-256: `2b899582c34e853dadd4be17dbb5fcd347e719d8d421d0afe516a200531b95c0`.

The original, first improved, round5 combined, and both round6 upscale TFLite files were verified unchanged after the experiment. The complete preservation hashes appear in `evaluation/result.json`. All candidates branch independently from the round5 parent, rather than accumulating edits from other candidates.

## 1. Rank channels using separate calibration probes

Calibration used 64 standard-normal latent vectors with seed 88900 and 64 uniform `[-1,1]` vectors with seed 88904. These inputs were separate from development and independent evaluation. Rankings were frozen before evaluating candidate edits.

We inspected feature maps at 32×32, 64×64, and 128×128. At 64×64, the existing calibrated phase correction was included. Per channel, we calculated:

- Activation standard deviation.
- Broad gradient magnitude after a reflected 5×5 box filter, with an interior crop.
- Two-pixel phase variation of the high-pass residual. Phase means were calculated per image before their RMS values were averaged, preventing cancellation across images from being the only ranking signal.
- An absolute downstream weight-path sensitivity estimate: multiply absolute pointwise color matrices, and propagate the resulting sensitivity backward through phase-averaged absolute projection matrices.

The sensitivity estimate ignores activation derivatives, cancellation, and input dependence. It is a heuristic for channel selection, not an exact gradient or a semantic interpretation.

Let `P` be phase variation, `E` broad gradient energy, `S` activation standard deviation, and `C` the path sensitivity estimate. Artifact ranking used descending `P*C`. Structure ranking used descending `E*C / (1 + 10*P/max(S,1e-8))`.

These scores do not identify an “eye channel” or a “hair channel.” They identify channels with particular measured responses on generated calibration inputs. Full scores and rankings are saved in `evaluation/channel_rankings.json`.

## 2. Apply a finite set of direct edits

We declared one unchanged baseline and 32 edited configurations before evaluation. Each configuration edits the top `k ∈ {1,4,8,16}` channels, with phase coefficient `b ∈ {0,0.1,0.25}` and detail coefficient `d ∈ {0,0.1,0.25}`, excluding duplicate unchanged settings.

**Selective phase balancing:** for selected output channels in the 32→64 and 64→128 spatial projections, replace each of the four phase filters with:

```
W_phase_new = (1 - b) * W_phase + b * mean_over_four_phases(W_phase)
```

The four-phase mean is conserved up to float32 rounding. Unselected output channels are unchanged.

**Selective feature residual:** use the existing zero-initialized residual convolution at 32×32. For selected structure-ranked channels only, set its channel-diagonal 3×3 kernel to center `d` and four axial neighbors `-d/4`. Every other coefficient remains zero. The branch's inherited first convolution, leaky activation, biases, and zero-padding convention are retained. It boosts existing feature variation; the coefficients contain no learned facial texture.

Every candidate's three edited tensors, configuration, checksums, metrics, first four generated portraits, and latent inputs were saved in checkpoints `001` through `033`. Checkpoint `000_channel_ranking` freezes calibration rankings and the plan. No candidate updates another candidate.

## 3. Freeze selection before independent evaluation

Development used 96 Gaussian noise inputs, seed 88901. We retained round7's criteria and added safeguards against shifting artifacts into other grid periods or merely cancelling them across images:

| Criterion | Required relative to parent |
|---|---|
| Fixed four-pixel phase RMS | ≤95% |
| Smoothed edge energy | ≥100.5% |
| Raw edge energy | ≥98% |
| Reflection-invariant nearest shape distance | ≥98% |
| Paired RGB RMS change | ≤0.015 absolute |
| Maximum RGB channel mean shift | ≤0.005 absolute |
| Clipped fraction increase | ≤0.002 absolute |
| Fixed two- and eight-pixel phase RMS | ≤105% each |
| Mean per-image four-pixel artifact RMS | ≤100% |

Smoothed edge energy is measured with a fixed binomial 5×5 filter, as in round7; it is distinct from the box filter used for channel rankings. Neither raw nor smoothed edge energy measures recovered facial detail. Coarse shape distances do not establish identity diversity.

Exactly one candidate passed every development criterion: **eight channels, phase coefficient 0.25, detail coefficient 0.25**. Its fixed four-pixel grid metric fell 8.36% and its smoothed edge metric rose 0.86%. It was frozen in `evaluation/frozen_selection.json` before evaluating new seeds. No channels or coefficients were changed afterward.

Selected zero-based channels:

| Operation | Channels |
|---|---|
| 32px detail residual | 52, 46, 45, 30, 3, 28, 61, 33 |
| 32→64 phase balancing | 17, 60, 52, 31, 18, 62, 16, 14 |
| 64→128 phase balancing | 47, 62, 56, 31, 40, 7, 43, 46 |

## 4. Independent results

The frozen edit was evaluated on 256 standard-normal inputs with seed 88902 and 256 uniform `[-1,1]` inputs with seed 88903. These generated samples are test probes, not ground-truth targets.

| Metric change versus preserved parent | Gaussian | Uniform |
|---|---:|---:|
| Fixed four-pixel grid RMS | −8.91% | +5.37% |
| Mean per-image four-pixel grid RMS | −4.00% | +0.055% |
| Smoothed edge energy | +0.87% | +0.93% |
| Raw edge energy | −1.51% | −0.54% |
| Nearest reflected-shape distance | +0.21% | +0.28% |
| Paired RGB RMS change, absolute | 0.00652 | 0.00423 |
| Maximum RGB mean shift, absolute | 0.00154 | 0.00093 |
| All predefined criteria passed | Yes | No |

Uniform noise failed the fixed four-pixel and per-image four-pixel artifact criteria. The cohort phase metric and per-image metric behave differently: a small change in individual patterns can change their batch average more substantially. We report both rather than treating the cohort metric alone as perceptual quality.

We inspected both eight-row comparisons in `figures/gaussian_comparison.png` and `figures/uniform_comparison.png`. Changes in edge definition and texture are subtle; inherited grid patterns remain. These comparisons do not establish clearer anatomy, realistic added texture, preserved identity, or fewer repeated identities. They support treating this as a modest experimental edit with noise-distribution-dependent behavior.

## 5. Exact weight and graph audit

Only three inherited weight tensors changed:

| Tensor | Changed float coefficients |
|---|---:|
| `base.upsamplers.3.projection.weight` | 2,048 |
| `base.upsamplers.4.projection.weight` | 2,048 |
| `base.refiner.conv2.weight` | 40 |
| **Total** | **4,136** |

All other state tensors are exactly unchanged. Within the two projection tensors, every unselected output channel is exactly unchanged. Selected channels' four-phase means changed by at most `2.98e-8` and `1.49e-8` respectively. Independent decoding of the exported 128px TFLite file found changes only in the corresponding three constant tensors. See `evaluation/edit_audit.json`.

The 128px export uses the existing graph architecture, with 39 operations and a 31,540,512-byte file. The 256px export appends round6's same analytic bilinear convolution/depth-to-space head, with 42 operations and a 31,542,608-byte file. Its enlargement remains interpolation; the selective feature edits happen before RGB output.

Both exports were compared with the edited Torch module using decoded TFLite execution with Torch arithmetic and independent NumPy arithmetic. Tests used zero, Gaussian, and uniform inputs and dynamic batches 1, 3, and 2. Maximum errors were `5.67e-7` for the 128px graph and `4.77e-7` for the 256px graph. A separate CLI smoke check generated four 256px portraits successfully.

**Native LiteRT acceptance is still unverified.** `validate_native.py` exited 2 with `NOT RUN` because neither a native LiteRT nor TensorFlow interpreter is installed. Our decoded-graph numerical checks are not native runtime validation. The supplied helper tests both exports at batches 1/2/3 once a native interpreter is available.

## Deliverables and status

Both files are explicitly named experimental because the frozen edit failed the cross-distribution promotion criterion:

```
candidates/model_selective_experimental.tflite
SHA256 5f899d68b63bf194197eb0d1dcf1e98f7002d6f0b8340f816e2f20b1132e0539

candidates/model_selective_experimental_2x.tflite
SHA256 0ff271e62cd7ed864d194d41efb574e7a22252e1900e36e3973b618621738977
```

The full edited Torch checkpoint is beside them. No previous model was replaced. The current application's exact noise distribution has not been established beyond “random noise”; therefore Gaussian-only success is not sufficient to promote this edit generally.

Generate a reproducible comparison sample using the decoded graph:

```sh
python3 round6/sample.py --model round8/candidates/model_selective_experimental_2x.tflite --count 8 --seed 88911 --output /tmp/selective_portraits.npz
```

The command writes latent vectors, RGB float images, and a matching PNG. Choose an unused output filename. This helper uses standard-normal noise. Native acceptance in an environment with a compatible interpreter:

```sh
python round8/validate_native.py
```

`selective_detail.py` contains the complete experiment and refuses to overwrite an existing plan. Reproducing the whole experiment requires the preserved parent, graph helpers from prior rounds, and Torch/NumPy/FlatBuffers. The PNG helper uses the workstation's existing Pillow fallback. Checkpoint `034_delivery` freezes the selected export state before visual review; `035_documentation` records the final review, audits, write-up, and file hashes. `logs/events.jsonl` records the run in UTC.

This experiment provides a better Gaussian-noise tradeoff than the prior uniform channel edits under the declared metrics. It does not establish a general improvement, learned super-resolution, or recovery of missing facial information. No thresholds were relaxed and no additional coefficient tuning was performed after inspecting the independent results.
