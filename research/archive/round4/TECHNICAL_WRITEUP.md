# Exporting and refining the restructured portrait generator

We continued from the smaller, function-preserving prototype and exported a new
TFLite graph. A calibrated feature-offset layer reduced a recurring grid
measurement on unseen inputs while leaving coarse output variation essentially
unchanged. The result is an experimental 31.5 MB TFLite file, compared with the
74.2 MB first improved model. All previous models and checkpoints are preserved.

This round did not train new identities, recover a training dataset, or learn a
new head angle. It tests whether specific structural additions can be initialized
usefully from the existing network and its own activations. Native LiteRT loading
remains an outstanding acceptance check.

## Starting point and preserved behavior

The earlier [restructuring experiment](../round3/RESTRUCTURING.md) replaced the
unused portion of the first transposed convolution with a dense projection,
expressed later upsamplers as channel projections followed by pixel shuffle,
and added a zero-initialized two-convolution residual branch at 32×32. Its
7,618,339 parameters include 73,856 new branch parameters. It preserves the
decoded first improved model to within 1.49 × 10^-6 on 273 test inputs.

The frozen parent checkpoint is
`round3/checkpoints/001_compact_identity/model_compact_identity.pt`, SHA-256:

```text
73c66d02ae1a51b8cbf2d62bd9bb189a71c41f5ea8a862e1d960cb1919688777
```

In this round, all candidates inherit those same weights. We neither overwrite
the first improved TFLite model nor the earlier repetition-focused v2 variant.
The original latent prior remains unknown. Gaussian and uniform noise are
explicit experimental input assumptions.

## Test spatial refinement rather than assuming it helps

After 4×4, the inherited synthesis path uses non-overlapping 2×2 upsampling and
pointwise color transforms. It has no convolution that mixes neighboring feature
cells at those finer scales. The added residual branch supplies that capacity,
but its initial zero output does not make the pictures better.

We programmed the branch with a small channel-diagonal neighbor correction:

```text
h_new = h + s × [neighbor_average(LeakyReLU(h)) - LeakyReLU(h)]
```

The first convolution is a center identity; the second has center weight `−s`
and four axial neighbor weights `s/4`, separately per channel. Its zero-padding
boundary behavior is part of this candidate, not a claim of perfect constant-field
preservation at the border. We tested `s = 0.025`, `0.05`, and `0.1`.

Smoothing alone reduced the dominant four-pixel phase metric by only 0.8–3.1% on
development inputs, while edge energy fell 0.9–3.6%. It failed our minimum 10%
artifact-reduction guard. We therefore retained the general spatial branch in
the exported structure with zero output. This experiment did not establish useful
hand-programmed weights for that branch.

## Calibrate an explicit phase-offset layer

The successful addition was more specific. We measured systematic feature
offsets associated with spatial phase at 32×32 and 64×64. For feature tensor `h`,
we computed a high-pass residual using a reflected five-pixel box average:

```text
r = h - box_average_5x5(h)
mu[c, dy, dx] = mean of interior r at row%2=dy, column%2=dx
offset[c, dy, dx] = -mu[c, dy, dx] + mean(mu[c, :, :])
h_new = h + strength × periodic_tile(offset)
```

The offset has zero spatial-phase mean per channel. It removes part of a measured
periodic component instead of smoothing every local edge. This is an additive
calibration from generated activations, with no optimizer or external image data.
The high-pass boundary uses reflection; a four-cell border is excluded from
calibration. A 2×2 correction in 64×64 feature space affects the four-pixel period
after the last upsampling to 128×128.

Calibration used 128 Gaussian vectors from seed 88500. Candidate selection used
a separate 128 Gaussian vectors from seed 88501. Before evaluating candidates,
we recorded both held-out seeds and these development guards:

| Guard | Requirement |
| --- | --- |
| Four-pixel phase RMS | At least 10% lower |
| Edge energy | Retain at least 95% |
| Reflection-invariant nearest shape distance | Retain at least 95% |
| Paired RGB RMS difference | At most 0.03 on the 0–1 range |
| Largest channel mean shift | At most 0.01 |
| Clipped-pixel fraction | Increase by no more than 0.002 absolute |

Eleven configurations included a baseline, the three spatial filters, three
64×64-only offset strengths, three two-stage offset strengths, and one combination.
Each received a small patch checkpoint, hash, full development metrics, and the
first 16 images and latent vectors in seed order.

| Selected development comparisons | Four-pixel metric change | Edge energy change |
| --- | ---: | ---: |
| Spatial branch at strength 0.1 | −3.1% | −3.6% |
| 64×64 offset at strength 0.25 | −19.7% | −1.2% |
| 64×64 offset at strength 0.5 | −36.8% | −1.8% |
| 64×64 offset at strength 0.75 | −48.2% | −1.9% |
| Offsets at 32×32 and 64×64, strength 0.75 | −48.2% | −2.0% |

We selected the single 64×64 offset at strength 0.75. The second offset stage
gave negligible additional benefit. The selection and its rationale were frozen
in checkpoint 012 before examining the held-out inputs.

## Rebuild the compact TFLite graph

The third-round prototype was a PyTorch checkpoint. This round implements a
FlatBuffer exporter that rebuilds the graph and all constant buffers, rather
than embedding the old file and retaining its unused weights. It uses standard
fields from the [official TensorFlow Lite schema](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/lite/schema/schema.fbs).

The graph contains a fully connected compact stem, reshape, the preserved early
transposed convolution, explicit upsampling projections, `DEPTH_TO_SPACE`,
leaky ReLU, pointwise convolutions, residual additions, and sigmoid. The selected
phase correction is an `ADD` after the 64×64 activation. The general 32×32 branch
is retained with its final convolution zeroed.

One transfer detail is important: TensorFlow NHWC `DEPTH_TO_SPACE` and PyTorch
`PixelShuffle` order phase channels differently. The exporter explicitly reorders
the weights from output-channel/row-phase/column-phase to
row-phase/column-phase/output-channel. The TensorFlow behavior is documented in
[tf.nn.depth_to_space](https://www.tensorflow.org/api_docs/python/tf/nn/depth_to_space).
The compact dense output is likewise reordered from NCHW flattening to NHWC
before reshaping. The earlier asymmetric crop is preserved rather than silently
changing head position.

| Export | Operations | File size |
| --- | ---: | ---: |
| Previous improved TFLite model | 54 | 74,217,344 bytes |
| Compact identity TFLite graph | 38 | 30,491,728 bytes |
| Compact graph with phase correction | 39 | 31,540,512 bytes |

The corrected export is **57.50% smaller** than the previous improved TFLite file.
Most of the reduction removes inaccessible weights; it is not quantization or
lossy approximation. The 64×64 offset is serialized as a broadcastable full
feature tensor, adding roughly one megabyte. Its underlying periodic calibration
has only 256 distinct channel/phase values. No speed or energy improvement has
been measured.

The exported contract remains float32 input `[batch, 512]` and float32 RGB output
`[batch, 128, 128, 3]`. The `serving_default` signature aliases remain `input_38`
and `activation_27`. A shape/slice/pack path maintains the batch dimension through
the early transposed convolution. We checked decoded batches of sizes 1, 2 and 3.

## Validate the actual exported graph

We wrote `check_graph.py` to read and execute the operations from the serialized
file. This avoids validating an export only through the module that generated it.
Both Torch arithmetic and a separate NumPy implementation were compared with the
module on zero, Gaussian and uniform inputs. The implementations still share the
FlatBuffer reader and our interpretation of the operation semantics.

We then executed the serialized graph on all 1,024 held-out vectors. Its maximum
disagreement with the module was **1.19 × 10^-7**. Thus the measurements below
represent the exported graph under the decoded semantics, not just a Python-only
variant. This does not establish native runtime compatibility.

`validate_native.py` is a separate acceptance script for an installed LiteRT or
TensorFlow interpreter. Here it exited with status 2 and a NOT RUN message because
neither runtime is available. It was not recorded as a passing native check.
Native loading, delegate behavior, and platform latency remain unverified.

## Measure the frozen candidate on unseen inputs

The held-out evaluation used 512 Gaussian inputs from seed 88502 and 512 uniform
inputs from seed 88503, paired between the inherited compact baseline and the
refined model.

| Change relative to the first improved model via its compact equivalent | Gaussian | Uniform from −1 to 1 |
| --- | ---: | ---: |
| Two-pixel phase RMS | −1.10% | −0.09% |
| Four-pixel phase RMS | **−52.35%** | **−36.53%** |
| Eight-pixel phase RMS | −50.39% | −34.70% |
| Edge energy | −1.99% | −0.90% |
| Reflection-invariant nearest shape distance | +0.15% | +0.26% |
| Shape effective rank | +0.16% | +0.11% |
| Paired RGB RMS difference | 0.00772 | 0.00906 |

These are further reductions relative to v1, not percentages relative to the
original unedited model. The shape-variation changes are effectively unchanged
for interpretation; we do not present them as a diversity improvement. The same
portrait family and some grid defects remain.

![Sequential Gaussian held-out comparisons](figures/gaussian_before_after.png)

Each pair shows the compact v1 baseline on the left and the refined output on
the right, using the same latent vector. The first 16 held-out vectors are shown
without selecting attractive examples.

![Sequential uniform held-out comparisons](figures/uniform_before_after.png)

The phase metric measures phase-averaged high-pass residuals in the image interior.
It targets systematic periodic response, rather than all possible grid defects.
The edge metric is mean absolute neighboring-pixel difference. Shape descriptors
are 16×16 pooled RGB images, mean-centered per channel and normalized for contrast;
reflection-invariant distance matches against either orientation. These are
engineering proxies, not human aesthetic preferences, face identities, or pose
labels. No reference image dataset or FID/KID evaluation was used.

## What this establishes and what remains

The structural transfer can be exported into a much smaller TFLite-format file,
and a small calibrated layer can reduce a targeted periodic artifact on new
inputs. General spatial capacity did not automatically produce useful guessed
weights: our simple smoothing program failed the selection guard and was left
inactive. That negative result is part of the experiment.

New identity coverage and learned head-angle control remain unestablished. We
have not recovered the real latent input protocol, original encoder, training
images, or paired multi-view data. Adding an ignored angle input or training on
more teacher-generated portraits would not establish those abilities. The next
data-backed stage is specified in `DATA_REQUIREMENTS.json`: recover real input
vectors, audit distinct identities and pose coverage, keep duplicates and identity
groups out of both training and test splits, and choose a learning objective that
matches the actual model use case.

The earlier v2 repetition-focused model is preserved as a separate branch. This
round does not combine its latent rebalancing with the new phase correction; that
combination would be a new candidate requiring separate calibration and tests.

## Checkpoints and reproduction

The new files are confined to `round4/`. Checkpoints 001–011 contain development
patches and previews; checkpoint 012 records the frozen choice. Timestamped
events and stdout are under `logs/`; all metrics and declared selection guards
are under `evaluation/`. Earlier models and publication artifacts remain intact.

```sh
# In a fresh experiment copy with the preserved compact parent available
python round4/refine_structure.py
python round4/finalize.py

# Preview the serialized graph using its decoded arithmetic backend
python round4/sample_compact.py \
  --model round4/candidates/model_compact_refined.tflite

# Validate the selected serialized graph against its module and NumPy arithmetic
python round4/check_graph.py \
  round4/candidates/model_compact_refined.tflite \
  --patch round4/checkpoints/012_frozen_selection/edits.npz

# Native acceptance, when the environment has LiteRT or TensorFlow installed
python round4/validate_native.py
```

These scripts need the same NumPy/PyTorch/FlatBuffers environment recorded in the
earlier experiments; rendering uses Pillow. New checkpoints and exports refuse
to overwrite their existing files. `sample_compact.py` clearly labels its backend
as decoded graph execution. The older `generate.py` supports the old architecture
and should not be used for these newly structured exports.

The refined TFLite SHA-256 is:

```text
58cdf5e8d50cd6d5e305ef6331faa35703d4c52d821638b88cc2fcd8084b7f00
```

Its companion `model_compact_refined.pt` stores the explicit PyTorch state. The
compact identity TFLite export provides a useful baseline for native acceptance
before adopting the corrected variant. Both are experimental until native
acceptance succeeds.
