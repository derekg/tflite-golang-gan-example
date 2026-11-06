# Direct detail edits without conventional training or external images

The user clarified that conventional training and an external dataset are excluded. This experiment uses only the existing model's weights and its generated outputs. It performs a fixed, declared sweep of analytical edits: no gradients, optimizer, external images, pretrained restoration model, or adaptive coefficient fitting. Generated portraits are evaluation probes, not supervised targets.

## Parent and preservation

The parent is `../round5/candidates/model_clean_diverse.pt`, SHA-256 `2b899582c34e853dadd4be17dbb5fcd347e719d8d421d0afe516a200531b95c0`. All edits branch independently from this parent. Its Torch and TFLite files and both round6 upscale exports were verified unchanged after the experiment; hashes are recorded in `evaluation/result.json`.

## Two analytical edits

**Phase balancing.** In the last two spatial upsampling projections, reshape each output channel's four subpixel phase filters into a group. Replace each phase filter `W_p` with `(1-b) W_p + b mean_p(W_p)`. This preserves the phase-average filter in exact arithmetic, reducing differences between phases. Biases, latent mapping, color layers, and existing phase correction buffers remain unchanged. Coefficients: `b ∈ {0, 0.025, 0.05, 0.1}`.

**Feature-space detail residual.** The existing residual branch at 32×32 has a zero final convolution. Set its channel-diagonal final filters to a negative discrete Laplacian: center `d`, axial neighbors `-d/4`, all other weights zero. Its existing identity first convolution and leaky activation remain. Coefficients: `d ∈ {0, 0.01, 0.025, 0.05}`. This boosts existing feature variation; it does not encode learned eyelashes or hair texture. The existing convolution uses zero padding, so border behavior is retained and must be reviewed.

All 16 Cartesian combinations were declared before evaluation. Checkpoints `000` through `015` store the three edited weight tensors, coefficient definitions, metrics, checksums, and four generated preview images with their latent inputs. These patches require the preserved parent and helper modules. They are experimental candidates, not approved replacements.

## Fixed evaluation

The plan was frozen in `evaluation/plan.json`. Development: 64 standard-normal noise inputs, seed 88800. Independent evaluation: 128 standard-normal inputs, seed 88801, and 128 uniform inputs in `[-1,1]`, seed 88802. No output was treated as a ground-truth high-resolution portrait.

The criteria required all of the following relative to the parent:

- Four-pixel phase RMS at most 95% of baseline.
- Gradients after fixed binomial 5×5 smoothing at least 100.5% of baseline.
- Raw edge energy at least 98% of baseline.
- Reflection-invariant nearest shape distance at least 98% of baseline.
- Paired RGB RMS change at most 0.015.
- Per-channel mean drift at most 0.005.
- Clipped pixel fraction increase at most 0.002.

These are conservative numerical proxies, not a validated perceptual detail score. Raw edge energy counts some artifacts, and the smoothed gradient measure captures broad structure rather than newly recovered fine detail. Passing these checks would still require visual review. Selection was declared as lowest phase RMS among eligible development candidates, with coefficients frozen before independent evaluation.

## Results

No development candidate met all criteria. No new model was promoted or exported.

Representative development changes:

| Phase balance b | Detail d | Four-pixel grid proxy change | Smoothed edge change | Raw edge change |
|---:|---:|---:|---:|---:|
| 0 | 0.05 | +1.70% | +1.75% | +1.80% |
| 0.025 | 0.025 | −3.08% | +0.51% | −1.27% |
| 0.05 | 0.05 | −6.12% | +1.04% | −2.52% |
| 0.1 | 0 | −14.60% | −1.34% | −8.34% |

Increasing feature-space sharpness also strengthened phase artifacts. Phase balancing reduced development grid energy but softened edges. The `b=0.05, d=0.05` candidate narrowly missed the raw-edge retention criterion; that threshold was not relaxed after seeing the result. This failure does not prove that viewers would dislike it, only that it did not meet the predeclared joint criteria.

Because there was no eligible candidate, the plan selected the lowest-phase-RMS candidate (`b=0.1, d=0`) as a diagnostic only. It remained ineligible regardless of independent results:

| Distribution | Four-pixel grid proxy change | Smoothed edge change | Raw edge change |
|---|---:|---:|---:|
| Standard normal | −15.40% | −1.32% | −8.42% |
| Uniform `[-1,1]` | +1.20% | −1.36% | −7.99% |

The uniform result also shows that reducing one artifact proxy on Gaussian inputs did not generalize to both input distributions. The exact intended noise distribution remains unspecified. Visual review of `figures/holdout_comparison.png` showed subtle softening and persistent grid texture. Both columns use the same analytic 2× interpolation for equal-sized display; no new learned upscale operation is involved.

## Logs and implementation

`direct_detail.py` is the complete reproducible implementation. The run initially stopped after writing checkpoint 000 because a logging function argument collided with the candidate-name field. The argument was renamed; the run resumed with the original frozen plan, checking the existing checkpoint arrays and manifest for exact agreement before continuing. Coefficients, seeds, metrics, and criteria were unchanged. This correction is recorded in `logs/events.jsonl`.

The script refuses to rerun a completed experiment in place. Incomplete runs may resume only with an identical frozen plan. Checkpoint `016_delivery` preserves the implementation and result manifest. Checkpoint `017_documentation` preserves this write-up and final hashes.

No native runtime validation was attempted for these candidates because none was exported. The previous round's native TFLite acceptance limitation remains unchanged.

## Conclusion for the next experiment

Direct analytical edits are possible without conventional training or outside data, but this sweep did not demonstrate an improvement in facial detail. It established a measured sharpness-versus-artifact tradeoff. Further experiments could target individual channels or spatial frequency bands instead of uniformly editing every channel. Such edits should remain hypotheses tested on fixed latent inputs, with saved candidate weights and human comparisons. Neither this experiment nor the earlier interpolation head demonstrates recovery of missing high-resolution information.
