# Editing a TFLite portrait generator without retraining

We inspected an unfamiliar TensorFlow Lite model, identified its output behavior, and edited its weights directly to reduce periodic artifacts and local repetition. A second experiment produced a modest reduction in repeated variants, while retaining the same recognizable portrait family. We also exported an exact horizontal-mirror variant. That mirror changes the image's facing direction; it does not create a new three-dimensional view.

The practical result is a preserved first improvement, an optional second weight variant, separate mirrored models, and a sampler that can select a less repetitive batch from the model's existing outputs. The evidence supports targeted changes to a limited generator, rather than recovery of a general portrait model.

## The experiment and its constraints

The input artifact was a single 74,217,344-byte `model.tflite`. There was no architecture code, training dataset, encoder, latent sampler, or reference image set. The approach was inspired by [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/): analyze and modify the actual parameter tensors using reasoning and tools. Our experiment used forward passes and a small, explicitly specified sweep of edit strengths. It was not a zero-shot or weights-only identification benchmark.

All numerical changes were analytical. We used no backpropagation, optimizer, fine-tuning, learned reward, or fit to external images. Some corrections used statistics from the model's own generated activations, and candidate selection used generated outputs. Sample selection and horizontal mirroring are described separately from weight editing because they change inference behavior through different mechanisms.

The first round is reconstructed from retained scripts, metrics, logs, and images. Prospective timestamped logging and per-candidate checkpoints began in the second round. This distinction matters for an accurate account of the work.

## Step 1 Decode the model and establish its behavior

The file identifier was `TFL3`, and the embedded description was `MLIR Converted.` The operator graph contained shape operations, reshapes, ten transposed convolutions, nine leaky ReLUs, and a final sigmoid. We wrote a small FlatBuffer reader to enumerate tensors, operators, constant buffers, tensor shapes, and exact byte offsets. The layout follows the [official TensorFlow Lite schema](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/lite/schema/schema.fbs).

| Property | Observation |
| --- | --- |
| Input | Float32 `[batch, 512]` |
| Input tensor name | `serving_default_input_38:0` |
| Output | Float32 `[batch, 128, 128, 3]` |
| Output range | Sigmoid RGB values from 0 to 1 |
| Channels | 512 → 1024 → 512 → 256 → 128 → 64 → 64 → 64 → 32 → 16 → 3 |
| Spatial sizes | 1 → 2 → 4 → 8 → 16 → 32 → 64 → 128, then three per-pixel stages |
| Hidden activation | Leaky ReLU with negative slope 0.3 |
| Stored float parameters | 18,551,459 |

Architecture alone established that this was an image decoder or generator. Forward passes with standard-normal latent vectors revealed painted-looking portraits, recurring compositions, and a pronounced periodic grid. GAN training is plausible, but the file does not establish a training objective, dataset, or intended input distribution. Gaussian and uniform noise are experimental assumptions, not recovered training facts.

TensorFlow and LiteRT were unavailable locally, and a package installation attempt failed because network access was unavailable. PyTorch 2.9.1 and NumPy 2.4.1 were already installed in another local Python environment. We reconstructed the forward pass with PyTorch and wrote a second implementation using NumPy in NHWC layout and explicit scatter accumulation. Pillow rendered the image grids.

The two implementations agreed to at most approximately 6 × 10^-8 at the output on zero, Gaussian, and uniform inputs, for the original, first improved, and second variant models. This validates their numerical agreement under our decoded semantics. Both share the same FlatBuffer parser and interpretation of the graph. It does not replace a native LiteRT compatibility check, which remains outstanding.

## Step 2 Locate useful structure in the weights

The first transposed convolution has a `[1024, 5, 5, 512]` kernel. Its input is only 1×1, and its output is 2×2. Under the decoded SAME-padding convention, only spatial positions `[1:3, 1:3]` contribute to that output. The other 21 of the 25 spatial positions do not affect this forward pass.

We flattened the active block into a matrix `A` with 4,096 rows and 512 columns, and inspected `AᵀA`. In the original model, the eight strongest eigen-directions accounted for 74.3% of its squared singular-value energy. Concentrated linear sensitivity suggested a possible handle on output variety. It did not prove the cause of repeated identities: later nonlinearities, training coverage, and the latent prior can all matter.

We also inspected the stride-two 2×2 kernels at 32×32, 64×64, and 128×128. Their four subpixel phases can give different responses even to an approximately constant feature field. This provided a concrete mechanism to investigate the visible grid.

## Step 3 Make the first weight edits

### Rebalance latent sensitivity

For `AᵀA = V diag(λ) Vᵀ`, we modified the active first-layer block using:

```text
A_new = A V diag(c × λ^(-α/2)) Vᵀ
```

The scalar `c` first normalizes the total linear-response energy, then applies a specified gain. Small `α` values reduce domination by the largest singular directions. Gain alone is equivalent to scaling the input latent vector; the directional rebalancing adds a different effect. We preserved the unused first-layer positions.

### Correct uneven phase response

For each targeted upsampling layer, we estimated the mean incoming channel vector `μ` from 128 Gaussian calibration inputs, seed 7391. For output channel `o` and subpixel phase `p`, define:

```text
d[o,p] = dot(W[o,p,:], μ)
target[o] = mean over phases of d[o,p]
ΔW[o,p,:] = β × (target[o] - d[o,p]) × μ / dot(μ, μ)
```

This is a minimum-norm rank-one correction along the measured mean feature direction. It moves phase responses toward their common value while preserving incoming directions orthogonal to `μ`. In contrast, averaging entire phase kernels softened more image detail in the first exploratory sweep, so we preferred the targeted correction.

### Select and freeze the first result

The saved scripts contain three development sweeps: eight configurations for whole-kernel phase blending and gain, seven for targeted corrections and spectral edits, and six for final selection, including repeated baseline entries. Each sweep used the same 64 development latents from seed 42. The final first-round setting was `α = 0.1`, gain `1.15`, and phase correction `β = 0.6`.

That model changed 2,162,686 stored float coefficients, or 11.66%. Only four weight buffers changed; graph bytes, shapes, signatures, and metadata remained identical. It was exported as `model_improved.tflite`, here called **v1**.

On two first-round held-out sets of 512 inputs each, v1 reduced the dominant 4- and 8-pixel phase measurements by roughly 57–60% relative to the original. Nearest-neighbor shape distance increased 12.8% on Gaussian inputs and 9.2% on uniform inputs. Some detail softened; a smaller 2-pixel artifact component increased. Those measurements supported a targeted artifact improvement, not a general aesthetic claim.

## Step 4 Preserve v1 before investigating repetition and pose

Before the second round, we copied v1, all existing Python scripts, the README, and first-round results into `checkpoints/000_preserved_v1`. We recorded SHA-256 hashes for both source models. New models were written under `round2/candidates`; neither source model was overwritten.

Every second-round candidate received a compressed weight patch, source and patch hashes, hypothesis, metrics, first 16 images in seed order, and matching latent vectors. Timestamps and execution events were appended to `logs/events.jsonl`, with a narrative in `WORK_LOG.md`. Source snapshots record the exact script versions.

A logger argument collision stopped the first development run immediately after it saved the baseline checkpoint. We fixed the parameter name, retained the failed source version, and resumed from the completed checkpoint. No model results changed. This implementation failure is recorded alongside the numerical experiments.

## Step 5 Test repetition and pose hypotheses

The second round used 128 Gaussian development inputs, seed 88421. Fifteen configurations, including the baseline, tested four kinds of edits.

| Family | Configurations | Development observation |
| --- | --- | --- |
| First-layer bias | Multiply by 0.9, 0.75, or 0.5 | Small changes in brightness, contrast, and local variation; same archetypes |
| Second-layer bias | Multiply by 0.9 or 0.75 | Little structural diversity benefit |
| Additional spectral rebalance | `(α, gain)` of `(0.1, 1.1)`, `(0.2, 1.2)`, `(0.3, 1.3)` | Higher nearest-neighbor distance and shape rank; stronger settings compress average separation |
| Combined edit | Mild spectral edit plus first bias ×0.9 | Small gain without a compelling visual advantage |
| Coarse spatial reflection | Blend the first feature lattice with its reflection at strengths 0.25, 0.5, or 1 | Blurred or malformed faces rather than controlled head rotation |
| Kernel reflection | Full kernel reflection or a special phase swap | Displaced or imperfectly reflected images |

We traversed the eight leading latent directions at coefficients −3 through +3, with other coordinates zero. These traversals mostly traded between recurring portrait compositions, illumination, and color. We did not identify a clean direction that changed only head yaw. The zero-centered traversal is a diagnostic slice of the network, not a representative random sample.

![Leading latent direction traversals](figures/principal_direction_traversals.png)

The coarse and kernel-reflection edits were rejected after visual review. The preserved contact sheet shows their failures as well as the numerical candidates.

![Development candidates](figures/development_contact_sheet.png)

## Step 6 Freeze the second weight variant and evaluate unseen inputs

We selected the additional spectral edit `α = 0.2`, gain `1.2`, applied to v1, before evaluating new inputs. The selection record is timestamped in checkpoint 016. These values are relative to v1; they are not the total edit relative to the original model.

This **v2** variant modifies 2,097,152 coefficients in v1's active first-layer block. All other bytes are unchanged. Its relative L2 change is 29.4% when measured over the entire stored first-layer kernel, including its unused positions. It is an optional experimental variant rather than a replacement for v1.

The strongest eight input directions account for 74.3% of active first-layer energy in the original model, 64.3% in v1, and 42.2% in v2. This establishes a change in linear sensitivity. Output metrics are needed to assess the behavioral effect.

We used 512 new Gaussian inputs from seed 88422 and 512 new uniform inputs from seed 88423, matching each latent vector between v1 and v2.

| Metric relative to v1 | Gaussian | Uniform from −1 to 1 |
| --- | ---: | ---: |
| Reflection-invariant nearest shape distance | +12.0% | +6.3% |
| Shape effective rank | +12.8% | +10.7% |
| Average shape distance | −9.1% | −11.3% |
| Average low-resolution pixel RMS distance | −1.3% | −0.2% |
| Edge energy | +1.6% | +3.9% |
| 4-pixel phase artifact RMS | +1.6% | +3.3% |

At a fixed reflection-invariant cosine-distance threshold of 0.05, the fraction of images with a close neighbor fell from **14.06% to 7.42%** for Gaussian inputs, and from **14.45% to 10.16%** for uniform inputs. These are descriptor-neighbor rates, not human-judged identity duplication rates.

To inspect sensitivity to batch composition, we also evaluated eight sequential blocks of 64 samples per distribution. All eight Gaussian blocks improved on nearest shape distance, while only four of eight uniform blocks improved. These are descriptive checks, not independent statistical replications.

The resulting distribution is less locally repetitive by these metrics, while its average shape separation is smaller. The images remain variations on the same portrait family. We therefore cannot claim that v2 learned new identities, expanded training coverage, or improved every aspect of image quality.

![Sequential held-out triplets with v1 then v2 then v2 mirrored](figures/normal_v1_v2_mirror.png)

Each triplet above uses the same latent input: v1, v2, then a horizontal mirror of v2. The first 16 held-out inputs are shown in order. The uniform-input comparison is saved separately.

## Step 7 Explain why reversing kernels did not give a clean mirror

Horizontal reflection commutes neatly with some convolution configurations, but this graph includes a 3×3, stride-two transposed convolution that expands 2 pixels to 4. Its full support is 5 pixels wide; the decoded SAME convention crops one pixel on the right and none on the left. The official [reference transposed-convolution implementation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/reference/transpose_conv.h) and [padding calculations](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/padding.h) describe the scatter and cropping behavior.

In one dimension, the full operation places input `x[i]` at output indices `2i + k`. Reversing both input positions and the three filter taps gives a reflection around index 2, whereas reflecting the four retained output pixels requires a reflection around index 1.5. The one-cell discrepancy at 4×4 expands through five further upsamplings, corresponding to a 32-pixel interior displacement before boundary effects.

The naive reflected-kernel candidate had RGB RMS error **0.2254** against an exact mirrored reference on the development pool. Replacing the 3-tap reversal with a phase swap reduced error to **0.1020**, but visible distortions remained. We did not present either as a new viewpoint or an exact mirror.

## Step 8 Export an exact horizontal mirror as an explicit graph change

For reliable facing-direction reversal, we added a final `REVERSE_V2` operation on axis 2 of the NHWC output. The export preserves the original weights byte for byte, appends an axis tensor and mirrored output tensor, and updates the graph output and signature mapping. The existing signature aliases remain available; the raw output tensor is named `mirrored_output:0`.

The new FlatBuffer root references original immutable tables retained in an aligned tail. The exported files are 1,520 bytes larger than their source models. Each mirror variant was checked on 64 Gaussian inputs: decoded output was bit-for-bit equal to `source_output[:, :, ::-1, :]`. Independent NumPy reference comparisons also passed. Native LiteRT loading and delegate execution remain unverified for these graph exports.

This gives an exact image reflection with zero weight changes. It swaps apparent left and right, including illumination and asymmetric details. It does not synthesize the hidden side of a face or provide continuous yaw control. Such claims would require pose labels, suitable model capability, and a different evaluation.

Both v1 and v2 have separate mirror exports. A caller can select the ordinary or mirrored model; there is no new random orientation mechanism hidden inside either network.

## Step 9 Reduce repetition in a displayed batch

The optional `sample_diverse.py` provides an inference policy for applications that display multiple portraits. It generates a pool of 256 Gaussian inputs, initializes selection at a descriptor medoid, and greedily chooses 16 samples whose minimum distance to the selected set is largest. Distances match each portrait against both the ordinary and reflected version of the other portrait, so a flip cannot count as a new identity.

For the saved v1 demonstration, seed 88424, nearest shape distance increased from 0.3486 in the first 16 pool items to 0.5397 in the selected 16. This is an expected selection effect on the chosen descriptor and a single-pool demonstration, not an independent benchmark of model improvement. Selecting 16 from 256 costs 16 times the inference work of generating exactly 16 samples, before selection overhead. Farthest-first selection can favor defects, unusual colors, or atypical composition because it has no aesthetic or face-identity judge.

An optional flag mirrors half the displayed images. This balances reflection status, not measured head yaw: frontal or already oppositely facing faces need not yield equal left-facing and right-facing counts. Mirror-invariant diversity is identical before and after that display transformation.

![Diverse selected batch with half the images mirrored](figures/diverse_batch.png)

## What the metrics measure

For shape measurements, we average-pool RGB images from 128×128 to 16×16, subtract each image's spatial mean separately per channel, flatten, and normalize its L2 norm. Cosine distance between descriptors measures coarse spatial appearance with reduced sensitivity to global brightness, channel offsets, contrast, and fine grid noise. It remains sensitive to expression, composition, pose, and defects. It is not a face-recognition embedding.

The reflection-invariant distance is the minimum of distance to another image and distance to its horizontal reflection. Self-matches are excluded. Nearest-neighbor distance is averaged across samples. Effective rank is the exponential entropy of the centered descriptor matrix's squared singular values. Both nearest distances and neighbor rates depend on sample count, so comparisons keep counts matched.

The phase-artifact metric computes a five-pixel high-pass residual, removes an eight-pixel border, averages residuals by spatial phase, and reports RMS variation of those phase averages. It measures systematic periodic response; it does not capture all texture defects or facial detail. Edge energy and clipping are supporting diagnostics. No reference dataset, FID, KID, identity labels, or yaw estimator was used.

## Checkpoints and reproduction

Use the Python environment with NumPy 2.4.1, PyTorch 2.9.1, and FlatBuffers; rendering requires Pillow. The recorded environment used Python 3.11.10 for inference and a separate local Python for Pillow. `generate.py` documents its renderer fallback. Native deployment additionally needs a TFLite or LiteRT interpreter.

```sh
# Inspect a model without writing to it
python inspect_model.py

# On a fresh experiment directory with the exact source models available
python round2/bootstrap.py
python round2/run_experiments.py
python round2/freeze_and_evaluate.py

# Generate a display batch from the preserved v1 model
python round2/sample_diverse.py \
  --model model_improved.tflite --pool 256 --count 16 --balanced-mirror

# Preview a mirrored export
python generate.py \
  --model round2/candidates/model_v1_mirrored.tflite \
  --count 16 --output mirror_preview.png
```

The existing checkpoint directory is an audit record. Completed candidate checkpoints are reused by the development runner; the final-selection stage refuses to overwrite its frozen checkpoint. Re-run from a clean experiment copy when reproducing the selection and evaluation sequence. A recorded environment, identical model bytes, and matching seeds are required; numerical differences across library versions and hardware are possible.

| Artifact | Location |
| --- | --- |
| Original model | `../model.tflite` |
| Preserved first improvement | `../model_improved.tflite` |
| Full first-round snapshot | `checkpoints/000_preserved_v1/` |
| Optional second weight variant | `candidates/model_diversity_v2.tflite` |
| Exact mirror of v1 | `candidates/model_v1_mirrored.tflite` |
| Exact mirror of v2 | `candidates/model_diversity_v2_mirrored.tflite` |
| Frozen second selection | `checkpoints/016_frozen_selection/` |
| Candidate metrics | `evaluation/development_metrics.json` |
| Unseen-input metrics | `evaluation/heldout_metrics.json` |
| Narrative record | `WORK_LOG.md` |
| Timestamped events | `logs/events.jsonl` |

SHA-256 checksums:

```text
original  0074d74ac51f7a67361c2f881faeeebb591f7297c3878f972538b4c1cb95e79b
v1        38c2106d8fded23210427ca1c27e30ec73acc00887597f2a26c82090fbbe31c9
v2        a68770b9cb6469e0659f811e987ed0242570b24365f6f97b1145d353c13111b4
```

The original and v1 hashes were checked again after exporting the new models. They remain unchanged.

## Conclusions supported by this experiment

Direct weight inspection exposed a useful low-dimensional structure and a phase-response mechanism that could be edited without retraining. The first result reduced a dominant periodic artifact. The second reduced local descriptor repetition on the tested input distributions, with mixed effects on broader diversity measurements. Neither edit established new identity coverage or a general head-pose control.

An explicit graph mirror reliably provides the opposite image orientation. A separate pool-selection policy can make displayed batches less repetitive. Keeping these mechanisms and their evidence distinct makes the result reproducible and keeps the claims proportionate to what was actually measured.
