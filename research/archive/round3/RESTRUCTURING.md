# Restructuring the portrait generator while preserving its learned output

Adding layers is possible, and their initial weights can be chosen analytically.
The useful starting point is to preserve the current generator, remove redundant
storage, and add capacity where the architecture has a concrete limitation. A new
layer initialized this way provides room for improvement; it does not immediately
create new identities, unseen face angles, or a better training distribution.

This third round implements a structural prototype and proposes the next learning
steps. All earlier TFLite models and second-round publication artifacts are
preserved. The prototype is a PyTorch checkpoint, not a deployable TFLite export.

## What the existing architecture suggests

The model already has 18.55 million stored float parameters, but roughly 11.01
million are in first-layer spatial positions excluded by its current padding and
input shape. Increasing raw parameter count is therefore an unconvincing first
response to repeated portraits.

There is a more relevant structural limitation: after the 4×4 stage, the model
upsamples with non-overlapping 2×2 stride-two kernels, followed by three 1×1
stages. These operations transform and subdivide each incoming feature cell;
they do not convolve neighboring cells together. Features still share global
latent ancestry and weights, so this is not a claim of statistical independence.
It is a lack of explicit neighboring-cell refinement at finer resolutions.

The 512 input values are another unresolved issue. We have not established that
the model was trained on independent Gaussian noise. They could be encoded
features or samples from another prior. Incorrect inputs could contribute to the
limited-looking outputs. Recovering the app's input-generation code or original
encoder should precede claims that the problem is purely architectural.

## The implemented prototype

| Existing operation | Replacement | Initialization |
| --- | --- | --- |
| 5×5 transposed convolution on a 1×1 seed | Dense 512 → 4096, reshaped to 1024×2×2 | Copy the four active spatial weight slices exactly; broadcast biases |
| 3×3 stride-two transposed convolution to 4×4 | Retain the same operation and asymmetric crop | Copy existing weights and biases |
| Five non-overlapping 2×2 upsampling stages | 1×1 channel projection followed by pixel shuffle | Reorder existing phase weights exactly |
| No spatial refinement at 32×32 | Two 3×3 convolutions in a residual branch | First convolution center identity; second convolution and biases zero |
| Three final 1×1 transposed convolutions | Ordinary 1×1 convolutions | Copy their equivalent channel transforms |

The forward structure is:

```text
512 input values
  → compact dense projection → 2×2 features
  → preserved cropped transposed convolution → 4×4 features
  → projection and pixel shuffle → 8×8 → 16×16 → 32×32
  → NEW residual spatial refiner
  → projection and pixel shuffle → 64×64 → 128×128
  → existing per-pixel color transforms → sigmoid RGB
```

### The new weights are deliberate rather than arbitrary

The residual block computes:

```text
h_new = h + Conv2(LeakyReLU(Conv1(h)))
```

`Conv1` starts as a center-tap identity. `Conv2` starts with zero weights and zero
biases. Its contribution is therefore exactly zero at initialization, including
for negative incoming values. We do not put an additional activation after the
skip addition: doing so would change negative features and break the initial
function match. Inserting a plain identity convolution followed by another leaky
ReLU would have the same problem.

This follows the general idea of function-preserving expansion studied in
[Net2Net](https://arxiv.org/abs/1511.05641) and
[Network Morphism](https://proceedings.mlr.press/v48/wei16.html). The particular
compact transfer and residual initialization here are derived from this file's
tensor layout; they are not a replication of those papers' experiments.

Each old 2×2 kernel is exactly representable as a 1×1 convolution producing four
times as many channels, followed by a pixel shuffle. The phase order is output
channel, subpixel row, subpixel column. This makes phase behavior explicit without
claiming to cure it: the transferred upsamplers initially retain the same artifacts.

## Verified results

The source was the preserved first improvement, SHA-256:

```text
38c2106d8fded23210427ca1c27e30ec73acc00887597f2a26c82090fbbe31c9
```

The prototype has **7,618,339 parameters**, including **73,856 new refiner
parameters**. Compared with 18,551,459 stored float parameters in v1, that is a
**58.93% reduction**. Most of the reduction removes weights that cannot contribute
under this graph's input shape and padding, rather than approximating useful
weights. The first bias expands from 1024 to 4096 stored values; upsampling biases
remain separate to avoid duplicating them across phases.

The serialized PyTorch state is 30,482,397 bytes versus the 74,217,344-byte source
TFLite file. These are different serialization formats, so the parameter count is
the cleaner structural comparison. We have not measured latency or energy use;
pixel shuffle and the new convolutions may change runtime behavior by backend.

| Test inputs from seed 88430 | Count | Maximum absolute output error |
| --- | ---: | ---: |
| Zero | 1 | 1.19 × 10^-7 |
| Standard normal | 128 | 8.34 × 10^-7 |
| Uniform from −1 to 1 | 128 | 7.15 × 10^-7 |
| Standard normal scaled by 3 | 16 | 1.49 × 10^-6 |

All outputs were finite and passed `allclose` with absolute and relative tolerance
2 × 10^-5. The residual block was also tested independently on negative and
positive feature values and preserved them exactly. Saving and reloading the
state preserved prototype outputs exactly on a probe batch. These tests compare
to our decoded PyTorch implementation of v1; native LiteRT execution remains
unverified.

The prototype's pictures should look the same as v1. That is the intended result
of this initialization, and no quality or diversity gain is claimed. There are
273 forward-comparison inputs, not a training run or an image-quality benchmark.

![Matched v1 and restructured prototype images](figures/v1_compact_identity.png)

## The next structure I recommend

I would develop the compact prototype in stages, preserving a matched-output
checkpoint before activating each addition.

1. **Verify the real input distribution.** Obtain the app's input sampler,
   training code, encoder, or representative real input vectors. Repeat existing
   comparisons using those vectors. A latent-prior error needs a different remedy
   from insufficient synthesis capacity.
2. **Activate spatial refinement gradually.** Start with the implemented 32×32
   residual block. If real evaluation supports it, add the same zero-initialized
   branches at 16×16 and 64×64. Coarser blocks can change larger structures; finer
   blocks can refine edges and texture. Initially freeze inherited parameters,
   so experiments have a controlled baseline.
3. **Add pose conditioning at coarse resolutions when pose supervision exists.**
   Use a small angle-conditioning branch that supplies feature scale and offset
   at 8×8 or 16×16. Initialize scale to one and offset to zero, preserving the
   parent function while initially ignoring the new control. Actual controllable
   yaw requires examples and a learning signal; an angle input alone supplies no
   knowledge of the hidden side of a face.
4. **Consider a residual latent mapping layer.** Start with `z_new = z + M(z)`
   and zero the final projection of `M`. This offers a place to adjust the input
   distribution after we understand it. It is not evidence of identity and style
   disentanglement. Style-based generators provide a research precedent for
   scale-specific controls, learned during training:
   [StyleGAN](https://arxiv.org/abs/1812.04948).
5. **Unfreeze inherited coarse layers only when needed.** If missing identities
   or poses cannot be expressed by the frozen backbone, small new adapters may
   not suffice. Training must then modify earlier synthesis weights or use a
   different pretrained generator with the required coverage. Preserve the old
   outputs as a regression set without making teacher matching the only objective.

I would not immediately add self-attention, randomly initialized hidden layers,
or more noise channels. Those additions can alter output statistics or inject
texture, but this experiment gives no reason to expect them to supply new
identity or pose knowledge without learning.

## Data changes that could support a real improvement

We currently have a model artifact, not its training data. The existing file does
not expose a dataset that can be cleaned, balanced, or enlarged. A future data
pipeline should distinguish the following:

| Desired change | Evidence and data needed |
| --- | --- |
| Fewer repeated identities | More genuinely distinct identities, near-duplicate auditing, and identity-aware evaluation |
| More left and right facing portraits | Balanced real pose coverage, or explicit mirrored augmentation for appearance reversal |
| The same identity at different angles | Multi-view or identity-linked examples, or a suitable learned 3D prior, with identity consistency checks |
| Cleaner painting detail | Higher-quality examples and a reconstruction, perceptual, or adversarial objective appropriate to the model |
| Reliable evaluation | Hold out identity or source groups so duplicate crops do not cross training and test boundaries |

Mirrored augmentation can balance reflected appearance. It does not create new
three-dimensional views. Generating more images from the existing teacher mostly
supplies its existing portrait family; teacher-only distillation cannot be relied
on to add missing coverage.

The learning setup depends on what we recover. If this is a decoder and an encoder
or latent/image pairs exist, adapters can be learned using reconstruction and
appropriate feature losses. If we instead train it as a generator from unpaired
portraits, we need an explicit generative training objective, such as an
adversarial discriminator, and validation of the input prior. We should not assume
the original file was trained as a GAN.

For a GAN path with limited real data, adaptive discriminator augmentation is a
relevant established training option, not an inference-layer trick:
[Training Generative Adversarial Networks with Limited Data](https://arxiv.org/abs/2006.06676).
It does not replace missing identity or pose coverage.

Before claiming improvement, compare against v1 using fixed real input samples,
separate quality and identity/pose measures, reflection-invariant repetition
checks, and a regression set. Our coarse shape metrics alone are insufficient to
establish preserved identity or new yaw capability.

## Files and reproduction

- `restructure.py` defines the architecture, exact transfer, and validation.
- `checkpoints/001_compact_identity/model_compact_identity.pt` holds its state.
- `checkpoints/001_compact_identity/manifest.json` records hashes, parameter
  counts, tests and limitations.
- `logs/events.jsonl` is the timestamped third-round record.
- `WORK_LOG.md` records the design and implementation sequence.

From the workspace root, using the existing Python environment with PyTorch:

```sh
python round3/restructure.py
```

The command creates a new checkpoint and refuses to overwrite the existing one.
Reproduction should therefore use a fresh copy of the experiment directory with
the exact v1 source model available. No earlier model is overwritten.

To load the checkpoint:

```python
import torch
from round3.restructure import RestructuredGenerator

model = RestructuredGenerator()
state = torch.load(
    'round3/checkpoints/001_compact_identity/model_compact_identity.pt',
    weights_only=True,
)
model.load_state_dict(state)
model.freeze_inherited()  # Only the new refiner is trainable
model.eval()
with torch.inference_mode():
    images = model(torch.randn(1, 512))  # NHWC float RGB
```

No training has been performed. No TFLite export or native deployment test is
included in this round. The immediate result is a smaller, extensible equivalent
model and a concrete path for further edits or data-backed learning.
