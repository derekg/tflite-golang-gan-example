# A small spatial cleanup layer without training

We added a 3×3 channelwise filter before the portrait generator's final color layers and compared it with further direct weight edits. There was no conventional training, optimizer, backpropagation, external image dataset, or pretrained restoration model. All coefficients were specified analytically; generated portraits were evaluation probes.

The selected experimental layer gives modest cleanup relative to round8. On independent inputs, mean per-image grid energy fell 4.65% for Gaussian noise and 4.30% for uniform noise. Smoothed edge energy fell 0.40% and 0.43%. It did not meet every frozen criterion: fixed four-pixel grid energy improved 4.34% on uniform noise, short of the required 5%. The candidate therefore remains experimental and does not replace round8.

## Parent and new architecture

Parent: `../round8/candidates/model_selective_experimental.tflite`, SHA-256 `5f899d68b63bf194197eb0d1dcf1e98f7002d6f0b8340f816e2f20b1132e0539`.

The original, round5, and round8 model files were verified unchanged after export. The selected layer leaves all 36 inherited constant tensors exactly unchanged. It reroutes only the input to `color0`, inserting:

```
128×128×64 final upsampling activation
    → symmetric one-pixel edge padding
    → 3×3 depthwise convolution, stride 1, multiplier 1, valid padding
    → existing color0 / color1 / color2 / sigmoid
    → optional unchanged analytic 2× upscale head
```

Each feature channel has its own spatial filter, with no cross-channel mixing and no new activation. TFLite stores its filter as `[1,3,3,64]`; the Torch reference uses `[64,1,3,3]` and groups=64. One-pixel symmetric padding duplicates the edge sample. The operator and option fields follow the [official TensorFlow Lite schema](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/lite/schema/schema.fbs), using `MIRROR_PAD` and `DEPTHWISE_CONV_2D`.

## Identity initialization and analytical choices

Every channel initially has a center weight of 1, all other coefficients zero, and zero bias. This identity insertion matched the parent's output within `1.20e-7`. It is numerically function preserving to float32 rounding, rather than bitwise identical.

We then tested convex mixtures between identity and two spatial kernels:

- Cross-neighbor average: four axial neighbors, each weight 1/4, center zero.
- Binomial 3×3 average: outer product of `[1/4,1/2,1/4]` with itself.

For mixing strength `s`, the kernel is `(1-s)*identity + s*neighbor_kernel`. Coefficients are nonnegative and sum to one. The filter preserves constant features and introduces no overshoot within an individual feature channel. Subsequent nonlinear color layers mean this does not guarantee an identical RGB response or RGB mean.

The selected candidate uses the cross kernel on all 64 channels at `s=0.05`:

```
0        0.0125   0
0.0125   0.95     0.0125
0        0.0125   0
```

This modest all-channel blend performed more consistently under the declared checks than filtering only a few high-scoring channels. Isolated channel edits sometimes increased output grid metrics. A possible explanation is that they disturb cancellation between feature channels before nonlinear color projection; this experiment does not prove that mechanism.

## Calibration, finite sweep, and checkpointing

The plan was frozen before calibration or candidate evaluation in `evaluation/plan.json`. Calibration used 64 standard-normal vectors, seed 89000, and 64 uniform `[-1,1]` vectors, seed 89004. At the 128px feature map, channels were ranked by per-image four-phase high-pass variation multiplied by an absolute downstream color-weight-path sensitivity estimate. This is an artifact heuristic, not a semantic channel label or exact derivative.

The first 16 ranked channels were `47,50,57,14,56,21,62,1,31,6,43,15,10,3,48,34`. Full scores are in `evaluation/channel_ranking.json`.

The fixed sweep contained 42 configurations:

1. One unchanged baseline.
2. Thirty-two added-layer candidates: cross/binomial kernels × top 8/16/32/all 64 channels × strengths 0.05/0.1/0.2/0.35.
3. Nine weight-only controls: further phase balancing in the last two spatial projections, using the prior round's artifact channel rankings, top 8/16/32 channels × strengths 0.05/0.1/0.2.

Every candidate starts from the same round8 parent. No candidates accumulate changes from others. Checkpoint `000_identity` contains the identity graph and tests; checkpoints `001`–`042` contain the candidate configuration, edited kernels/projections, checksums, and full development metrics. Unchanged projection arrays are included in layer patches to make restoration explicit. Checkpoint `043_delivery` freezes the selected export before visual review, and `044_documentation` records the final audits and review.

Development used 64 Gaussian inputs, seed 89001, and 64 uniform inputs, seed 89005. Eligibility required all criteria on both distributions. Selection minimized the worst per-image grid ratio among eligible candidates. With no eligible candidate, the frozen fallback selected the diagnostic with fewest failed checks, then lowest worst per-image grid ratio. It selected `cross_channels64_strength0.05`; its only failed development check was uniform fixed-grid reduction. Selection was frozen before independent evaluation, with no subsequent coefficient changes.

## Cleanup criteria

These criteria were declared for this new cleanup experiment, before evaluation. Unlike round8's sharpening experiment, this round allows a small loss of edge energy in exchange for reduced texture. Criteria were not changed after viewing results.

| Metric | Requirement relative to round8 |
|---|---|
| Fixed four-pixel phase RMS | ≤95% |
| Mean per-image four-pixel artifact RMS | ≤97% |
| Smoothed edge energy | ≥99.5% |
| Raw edge energy | ≥94% |
| Reflection-invariant nearest shape distance | ≥98% |
| Paired RGB RMS change | ≤0.02 absolute |
| Maximum RGB channel mean shift | ≤0.005 absolute |
| Clipped pixel fraction increase | ≤0.002 absolute |
| Fixed two- and eight-pixel phase RMS | ≤105% each |

These are engineering proxies. Raw edges include some grid artifacts; smoothing can lower that measure for desirable and undesirable reasons. The smoothed edge measure captures broader structure, not genuine recovery of facial detail. Shape distances do not identify people or prove identity diversity. No high-resolution targets were available.

## Independent results

The frozen diagnostic was evaluated on 256 Gaussian inputs, seed 89002, and 256 uniform inputs, seed 89003.

| Change relative to round8 | Gaussian | Uniform |
|---|---:|---:|
| Mean per-image four-pixel grid RMS | −4.65% | −4.30% |
| Fixed four-pixel grid RMS | −6.49% | −4.34% |
| Fixed two-pixel grid RMS | −9.05% | −8.69% |
| Fixed eight-pixel grid RMS | −6.23% | −4.29% |
| Smoothed edge energy | −0.40% | −0.43% |
| Raw edge energy | −4.40% | −4.46% |
| Nearest reflected-shape distance | +0.002% | −0.018% |
| Paired RGB RMS change, absolute | 0.00182 | 0.00114 |
| Maximum RGB mean shift, absolute | 0.0000984 | 0.0000594 |
| All independent criteria passed | Yes | No |

The independent Gaussian checks passed, but neither that outcome nor the visually subtle cleanup overrides the frozen development and uniform failures. The file is named experimental, with no promotion or parent replacement.

## Size and numerical validation

The new filter stores 576 weights and 64 zero biases: **640 floats / 2,560 bytes**. Padding constants and graph metadata account for the remaining small file increase.

| Export | Bytes | Operations |
|---|---:|---:|
| Round8 128px parent | 31,540,512 | 39 |
| Round9 128px experimental | 31,543,712 | 41 |
| Round8 256px export | 31,542,608 | 42 |
| Round9 256px experimental | 31,545,776 | 44 |

The 256px model is **31.55 MB**, 3,168 bytes larger than round8. The slight difference between 128px and 256px file increments reflects FlatBuffer layout/alignment. The 2× enlargement head itself is unchanged.

Both exports were decoded and executed with Torch and independent NumPy arithmetic, comparing with the source module on zero, Gaussian, and uniform inputs at dynamic batch sizes 1, 3, and 2. The largest error across both exports/backends was `1.103e-6`. A separate random feature probe validated depthwise layout and borders within `2.385e-7`. A CLI smoke check produced four 256×256 images successfully.

The final tensor audit confirmed all 36 inherited constants are exactly unchanged, the only inherited operation reroute is `color0`'s feature input, and the added filter has exactly the declared coefficients. See `evaluation/edit_audit.json`, `identity_checks.json`, and `graph_validation.json`.

**Native acceptance remains unverified.** The native validation helper exited 2 with `NOT RUN` because this environment has neither LiteRT nor TensorFlow installed. Decoded execution validates our arithmetic interpretation, not native runtime/delegate acceptance. The helper is supplied for a compatible environment.

## Same-noise visual comparison

We reused the exact eight Gaussian vectors, seed 88920, and eight uniform vectors, seed 88921, from the user's original-vs-round8 comparison. All samples remain in input order, with none screened out. Each new saved archive contains latents plus original, round8, and round9 RGB arrays. Original and round8 outputs were checked against the prior saved archives: latent arrays match exactly, original output error is zero, and round8 differs by at most `1.50e-7` from the earlier decoded-graph evaluation.

All columns use the same analytic 256px bilinear enlargement with no per-image normalization, additional sharpening, or retouching. Files include:

- `figures/gaussian_round8_vs_round9_00_03.png` and `_04_07.png`.
- Corresponding uniform-noise sheets.
- `figures/gaussian_all_8.png` and `uniform_all_8.png`, showing original → round8 → round9.

We reviewed all 16 pairs. The filter makes a small texture change, with slight softening and persistent coarse grid structure. It does not fix the badly formed examples or visibly introduce new eyes/hair detail. There was no blinded human preference test, identity evaluation, or proof of recovered image information.

## Deliverables and reproduction

```
candidates/model_spatial_experimental.tflite
SHA256 4008ef0fd929fc45b6f49a6ecdf596fc39a9620754296132bea25e80a583c3c3

candidates/model_spatial_experimental_2x.tflite
SHA256 917f0548facfac627fefa3fec325eed89fe992890744b4ad63bef47c7cf8f273
```

The full Torch state is beside the 128px file. It includes the parent module and `spatial_kernel` buffer; reconstruct `SpatialGenerator` with `evaluation/result.json`'s chosen configuration and `evaluation/channel_ranking.json`, then load its state dict. It is a different state layout from the unwrapped round8 checkpoint.

Generate new Gaussian portraits with the decoded graph:

```sh
python3 round9/sample.py --count 8 --seed 89020 --output /tmp/round9_portraits.npz
```

This saves float32 outputs and noise inputs plus a PNG. Choose an unused filename. Native acceptance in a compatible environment:

```sh
python round9/validate_native.py
```

`spatial_cleanup.py` is the complete build/evaluation source and refuses to overwrite an existing plan. Full reproduction requires the preserved round8 files and earlier-round helpers. Rendering uses system Python/Pillow solely to arrange computed images. Candidate generation and independent numerical evaluation use the existing Torch/NumPy/FlatBuffers environment. All logs are in `logs/events.jsonl` with UTC timestamps. Prior models, checkpoints, and publication bundles are preserved.
