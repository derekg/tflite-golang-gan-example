# Reducing the portrait generator's grid without training

Round11 changes the final upsampling projection so neighboring feature cells contribute to the same output region. The weights are calculated from the existing model. No external images, backpropagation, optimizer, or conventional training were used. A finite set of coefficients was evaluated on generated probes.

On 512 fresh noise inputs, compared with round9, the selected model reduced normalized periodic edge strength by about 30% and repeating brightness texture by about 23%. Broad-edge energy decreased by 3.9–4.4%; raw-edge energy decreased by 18.1–18.5%. This is useful cleanup with some softening, not recovered facial detail. Existing malformed portraits remain malformed.

The experiment preserves the original, round8, round9, round10, and the existing browser page. The new model remains separately named `experimental` because native LiteRT and browser acceptance have not been tested.

## 1. Diagnose the grid in native output

The generator maps 512 floating-point noise values to a 128×128 RGB portrait. Its spatial stages grow through 2, 4, 8, 16, 32, 64, and 128 pixels. Five of the original layers use 2×2, stride-2 transposed convolution. Those kernels do not overlap. The compact implementation expresses the same operation as a 1×1 channel projection followed by pixel shuffle. The compact rewrite did not introduce this isolation.

After the final upsampling stage, three 1×1 color layers cannot exchange information between neighboring pixels. Earlier work added a small spatial refiner at 32 pixels and a weak depthwise mixer at 128 pixels, but these did not eliminate the texture.

We measured the original, round8, and round9 on the same 32 Gaussian and 32 uniform inputs. The new edge diagnostic computes horizontal and vertical absolute RGB differences, averages each over the perpendicular axis, and groups the resulting edge profiles by coordinate modulo 2, 4, 8, and 16. For each image and direction, the standard deviation between these phase groups is divided by mean edge magnitude. The headline score averages the 2-, 4-, and 8-pixel values across images and directions. A uniform reduction of all edge magnitudes therefore cannot, by itself, improve this score.

All measurements use native 128px output and an interior crop, before the optional 256px interpolation. Coarser 16px groups are reported as a diagnostic, not included in the selected objective. Portrait structure can affect these proxies, so they are paired with structure checks and actual image review.

| Noise law | Original edge-grid score | Round8 | Round9 |
|---|---:|---:|---:|
| Gaussian | 0.3011 | 0.2695 | 0.2773 |
| Uniform [-1,1] | 0.2907 | 0.2786 | 0.2869 |

Round9's absolute texture energy fell in prior tests, yet its normalized edge periodicity increased here. This explains why a small smoothing filter could improve one metric while leaving the visible grid conspicuous. Feature-space phase profiles were also recorded at each spatial resolution. Channel magnitudes and nonlinear color mixing prevent interpreting those profiles as direct measures of visible artifact severity.

This architectural hypothesis is consistent with [Odena, Dumoulin and Olah's investigation of deconvolution artifacts](https://distill.pub/2016/deconv-checkerboard/): learned phase-dependent filters can create artifacts even when kernel size is divisible by stride, and resize-based weight tying can discourage them. Their research motivates our test; it does not establish that this particular pretrained model can be repaired without tradeoffs.

## 2. Construct overlapping weights analytically

Let `Wp` be the existing channel projection for one of the four pixel-shuffle phases, and let `M = mean_p(Wp)`. Let `delta` be a 3×3 kernel with a one at its center. For phase `p`, `Bp` is the separable half-pixel bilinear interpolation kernel:

```text
even-axis taps = [0.25, 0.75, 0.00]
odd-axis taps  = [0.00, 0.75, 0.25]
Bp = outer(vertical phase taps, horizontal phase taps)

Wprime_p = (Wp - b*(Wp - M))*delta + a*M*(Bp - delta)
```

The two coefficients have different jobs:

- `a` makes the shared component continuous across parent-cell boundaries.
- `b` attenuates differences between the four phase-specific projections.

At `a=b=0`, this is the original projection padded with zeros. At `a=b=1`, it is bilinear interpolation of the phase-mean channel projection. The selected model uses `a=0.5`, `b=0.2` only at the final 64→128 upsampling stage.

For constant spatial input, the average response across phases is preserved algebraically because each interpolation kernel sums to one. Individual phase responses change when `b` is nonzero. Nonlinear activations and spatially varying inputs mean that average output RGB is not guaranteed to be identical; it is measured separately.

During the coefficient sweep, an equivalent efficient implementation evaluates the old projection and the bilinear shared component separately. Export folds the complete operation into one 3×3 projection kernel, with replicated-edge padding, followed by the inherited pixel shuffle, bias, and activation. This is an actual change to the generator's feature projection, before its color layers.

## 3. Round10: localize the useful intervention

Round10 froze a plan containing 23 configurations: baseline, individual interventions at output sizes 16/32/64/128, combinations across the later stages, a phase-only control, and three ordinary smoothing controls. Development used 32 Gaussian inputs from seed 89100 and 32 uniform inputs from seed 89101.

The structure-safe candidate with the lowest worst-distribution edge-grid ratio changed only the final upsampler, using `a=0.35`, `b=0`. It reduced the edge-grid score by about 36% on the independent 128+128 holdout, while brightness phase4 improved only 8–10%. It did not meet the predeclared 20% brightness-pattern reduction threshold, so it was saved as a diagnostic model rather than declared a complete solution.

The ordinary smoothing controls increased normalized grid strength on development. Large changes across earlier stages softened broad structure too much. Phase averaging without sufficient overlapping interpolation also performed poorly. These observations justified a narrower follow-up.

Each candidate has a weight patch, preview, and manifest in `../round10/checkpoints/`. A logger argument collision interrupted the first run after baseline evaluation; the logger was corrected and the baseline checkpoint restored. The frozen plan, formulas, seeds, and thresholds did not change. Both source snapshots and the resume event were retained.

## 4. Round11: tune continuity and phase balance separately

The follow-up branches directly from round9, not from the round10 diagnostic. It uses 16 predetermined combinations:

```text
a ∈ {0.35, 0.50, 0.65, 0.80}
b ∈ {0.00, 0.10, 0.20, 0.30}
changed stage = final 64→128 upsampler only
```

Fresh development inputs: 64 Gaussian, seed 89200; 64 uniform, seed 89201. The round10 guards were carried forward unchanged:

| Check, relative to round9 on identical inputs | Requirement |
|---|---:|
| Normalized edge-grid score | ≤85% |
| Mean per-image high-pass phase4 RMS | ≤80% |
| Binomial-smoothed edge energy | ≥95% |
| Reflection-invariant nearest shape distance | ≥97% |
| RMS after 8×8 averaging | ≤0.020 |
| Whole-image RGB RMS | ≤0.035 |
| Largest channel mean shift | ≤0.008 |
| Clipped-pixel fraction increase | ≤0.002 |

Both distributions must pass all checks. Among passing candidates, select the smallest worst-distribution edge-grid ratio. Only `a=0.5, b=0.2` passed on development. Selection was written to disk before evaluating holdout seeds. Stronger overlap settings reduced the grid further but failed structure preservation.

## 5. Independent results

Holdout: 256 Gaussian inputs, seed 89202, and 256 uniform inputs, seed 89203. None of these inputs were used for coefficient selection.

| Measurement versus round9 | Gaussian | Uniform |
|---|---:|---:|
| Normalized edge-grid score | −29.68% | −29.57% |
| Mean per-image brightness phase4 RMS | −22.54% | −22.92% |
| Broad-edge energy | −3.92% | −4.42% |
| Raw-edge energy | −18.12% | −18.54% |
| Nearest shape distance | −0.04% | −0.24% |
| Paired RGB RMS, range 0–1 | 0.01116 | 0.00708 |
| Paired 8×8-averaged RMS | 0.00347 | 0.00234 |
| Maximum channel mean shift | 0.00198 | 0.00166 |

All eight checks passed on both holdouts. The nearest-shape measurement suggests little change in this diversity proxy; it does not establish unique identities or improved anatomy.

Paired bootstrap intervals, 2,000 resamples of the held-out input pairs, put the edge-grid reduction at 29.47–29.88% for Gaussian noise and 29.30–29.82% for uniform noise. These intervals describe sampling variability of this metric under those noise laws, not uncertainty in human aesthetic quality.

Detailed 2/4/8/16-pixel edge-profile values, clipping, repetition metrics, and all guards are in `evaluation/holdout.json`. The model's intended noise distribution remains unknown, which is why both laws were checked.

## 6. Visual inspection

The comparison inputs from earlier rounds were reused exactly: first eight Gaussian vectors from seed 88920 and first eight uniform vectors from seed 88921. All are included, including poor outputs. Each model receives the same float32 vector; no candidate screening or image retouching is performed.

The PNGs show native 128px pixels enlarged 2× with nearest-neighbor display. This makes the texture visible and avoids hiding it with the 256px interpolation head. Edge panels use absolute RGB finite differences, a shared exposure derived from the round9 batch, and the same fixed gamma for both models. Edge panels are diagnostics; RGB panels have no contrast normalization.

Review of the 16 established comparison outputs shows less regular mesh-like texture across skin and backgrounds, with slightly softer transitions. Residual coarse blocks are visible. Gaussian samples #05 and #07 remain particularly poor. Fresh holdout examples also include distorted faces that cleanup does not repair.

- [Gaussian RGB and edge comparison, samples 00–03](figures/gaussian_edges_00_03.png)
- [Gaussian portraits, samples 04–07](figures/gaussian_portraits_04_07.png)
- [Uniform portraits, samples 00–03](figures/uniform_portraits_00_03.png)
- [Uniform portraits, samples 04–07](figures/uniform_portraits_04_07.png)
- [Fresh Gaussian holdout, samples 00–03](figures/gaussian_holdout_portraits_00_03.png)

A copied renderer initially retained an edge-column lookup for round10. It was corrected to round11 and all figures regenerated before delivery. Model outputs, selection, and numerical evaluation were unaffected. The final source-to-image audit verifies RGB and edge panel contents against their named arrays.

## 7. Model export and numerical checks

Only the inherited `upsample4/weights` tensor changes. It expands from 16,384 to 147,456 floats, adding 131,072 stored float values. All other inherited constants are checked for exact equality. The graph adds one symmetric padding operation; a one-pixel symmetric pad is equivalent to replicated edges here.

| Export | Bytes | Graph operations |
|---|---:|---:|
| Native 128px | 32,068,288 | 42 |
| Interpolated 256px | 32,070,368 | 45 |

The 256px file is about 32.07 MB, up 524,592 bytes from round9's 256px model. Its final interpolation head does not create missing high-resolution details. Expanding a dense projection from 1×1 to 3×3 increases arithmetic cost; the small file-size increase does not imply unchanged inference speed.

Validation includes zero-strength identity, random rectangular feature maps including boundaries, phase-mean DC preservation, and equality between the direct interpolation formula and the expanded kernel. Exported FlatBuffers are then parsed and executed with separate Torch and NumPy arithmetic implementations. Both resolutions are checked on zero, Gaussian, and uniform inputs, including dynamic batches of 1, 2, and 3. The largest export-versus-module difference was `8.65e-7` on the 0–1 scale.

These are decoded-graph checks. No native TensorFlow/LiteRT interpreter is installed here, and this new model has not been tested in the browser. The current browser artifact remains on its preserved round9 model.

Native 128px SHA-256:
`306f8796778eade5935903f74a901844ad385af0f445f052625ce31f2ebfeb81`

Interpolated 256px SHA-256:
`cc4ea748d98a8099fab3f8aefee913cb24ea6fb6cb4589922ece5b8d069dd6f5`

## 8. Reproduction and checkpoints

Run from the repository root with the Python environment containing Torch, NumPy, and FlatBuffers. Rendering uses system Python with Pillow. Existing completed experiment directories intentionally cannot be overwritten.

```sh
# Generate new samples from the actual exported FlatBuffer:
python3 round11/sample.py \
  --seed 90000 --count 8 --output /tmp/round11-new-samples.npz

# Regenerate comparison PNGs from saved outputs:
python3 round11/render.py

# On a machine with native LiteRT or TensorFlow installed:
python round11/validate_native.py
```

`../round10/overlap.py` contains the formula, diagnostic metrics, exporter, and broad experiment. `refine_overlap.py` contains the focused experiment. Both depend on the preserved repository helpers and round9 checkpoint. The publication bundle is an evidence/code package, not an independent replacement for the repository or a bundle of model weights.

`checkpoints/000` through `015` retain each candidate's edited kernel, preview, metrics, configuration, and checksum. Checkpoint `016_delivery` records the selected model and export hashes. The final documentation checkpoint also records the source and figure hashes, native-test status, visual findings, and preservation checks. Logs are append-only JSONL.

The remaining problem is both textural and semantic: weaker periodic seams do not solve coarse anatomy or infer details the original generator never represented. Further experiments should preserve this version, measure those two problems separately, and test on new seeds before changing the browser default.
