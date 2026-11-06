# Adding a 2× upscale head without retraining

We appended an analytically initialized RGB head to the existing improved portrait generator. The new TFLite output is 256×256 instead of 128×128. The input remains a batch of 512-dimensional float32 noise vectors. Every inherited constant tensor is numerically identical to its parent. No training data, optimizer, or gradient updates were used.

This is interpolation implemented inside the network. It smooths enlargement but does not recover missing facial detail, create identities, correct pose, or solve repeated portraits. The visible grid texture in the parent remains visible. A learned super-resolution model would require additional evidence or training that this experiment does not provide.

## Parent and preservation

Parent: `../round5/candidates/model_clean_diverse.tflite`.

Parent SHA-256: `a19b85ceffb1eb0f9f8c515cf5e6cfbd0acf5eda861086d9e7de393594de13c3`.

The exporter reconstructs the parent's graph from its saved graph specification and decoded constants, then appends the head. Tensor indices for all inherited tensors remain unchanged. It checks exact array equality for every inherited constant, including shape constants. Original model files are never overwritten; hashes of the original, first improved, and round5 models were verified after export. The full integrated new models are in `candidates/`; the tiny head initialization is separately saved in `checkpoints/001_analytic_head/weights.npz`.

## Architecture and calculated weights

The default adds three operations after the parent's sigmoid RGB output:

1. Symmetric one-pixel padding: `[N,128,128,3] → [N,130,130,3]`. For a one-pixel border, symmetric padding duplicates the edge sample.
2. Valid 3×3 convolution: 3 input channels → 12 phase channels, returning `[N,128,128,12]`.
3. Depth-to-space with block size 2: `[N,128,128,12] → [N,256,256,3]`.

The convolution stores 324 weights and 12 zero biases: 336 new float parameters, with 48 nonzero weights. It has no color mixing. Output channels are ordered `(phase_y * 2 + phase_x) * 3 + color` for TFLite NHWC depth-to-space. The PyTorch reference explicitly converts this DCR phase ordering to PixelShuffle's CRD ordering.

For half-pixel bilinear enlargement, output coordinate `u` samples input coordinate `(u + 0.5)/2 - 0.5`. At even output coordinates `2i`, the one-dimensional taps are `[0.25, 0.75, 0]` over input indices `[i-1,i,i+1]`. At odd coordinates `2i+1`, they are `[0, 0.75, 0.25]`. Each two-dimensional phase kernel is the outer product of its vertical and horizontal taps. All taps are nonnegative and sum to one. Consequently the default is a convex interpolation: it preserves constants and stays within the source pixel range without a clip operation.

These are calculated weights, rather than randomly guessed coefficients. The implementation agrees with PyTorch half-pixel bilinear interpolation to float32 rounding error. Operator encodings follow the [official TensorFlow Lite schema](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/lite/schema/schema.fbs); phase layout follows [TensorFlow depth_to_space](https://www.tensorflow.org/api_docs/python/tf/nn/depth_to_space).

## Optional sharpened export

The second export appends a channel-diagonal 3×3 sharpening convolution at 256×256. Its center is `1 + s`; its four axial neighbors are `-s/4`; `s = 0.1`. It uses the same edge padding, then clamps RGB to `[0,1]` with maximum/minimum operations. This adds another 84 convolution parameters and four operations, for 420 new float parameters and seven new operations in total relative to the parent.

The development comparison also evaluated strength 0.25, but only the conservative 0.1 variant was exported alongside the default. Sharpening makes a small visible difference and can emphasize existing artifacts. It has no demonstrated advantage in perceptual portrait quality.

| Model | Output | Operations | File bytes | Added bytes vs parent |
|---|---|---:|---:|---:|
| Existing improved parent | 128×128 | 39 | 31,540,512 | — |
| `model_upscale_2x.tflite` | 256×256 | 42 | 31,542,608 | 2,096 |
| `model_upscale_2x_sharp.tflite` | 256×256 | 46 | 31,543,952 | 3,440 |

## Experiment sequence

1. Freeze the parent checksum, architecture, candidate definitions, and seeds in `evaluation/plan.json` and `logs/events.jsonl`.
2. Save the analytic head weights as checkpoint 001 before evaluating them.
3. Compare nearest-neighbor enlargement, bilinear, sharpening 0.1, and sharpening 0.25 on 32 Gaussian development inputs, seed 88700.
4. Check random images, constants, ramps, and a diagonal impulse pattern against the independent interpolation operation, including border pixels and RGB channel handling.
5. Evaluate unchanged candidates on 64 Gaussian inputs (seed 88701) and 64 uniform inputs in `[-1,1]` (seed 88702). The user's noise distribution is not specified beyond random noise, so both were retained.
6. Render a fixed eight-portrait comparison. Each row uses identical latent input; columns show nearest-neighbor baseline, the default head, and optional mild sharpening. The baseline is the parent's original pixels enlarged to the same display size.
7. Export two integrated FlatBuffers, verify inherited constants, execute their decoded graphs with Torch arithmetic and independent NumPy arithmetic, and compare with the source module plus the head.
8. Check dynamic batches 1, 2, and 3, recheck protected parent file hashes, freeze delivery checkpoint 002, and add documentation/reproduction helpers in checkpoint 003.

## Measurements and their limits

There is no real 256×256 target. We used two numerical proxies: area-downsample the native image to 64×64 then enlarge back to 128×128, and enlarge native 128×128 to 256×256 then area-downsample it back. PSNR uses peak RGB value 1. These measure reproduction of the generator's current pixels, including its artifacts; neither is a perceptual quality benchmark.

| Holdout distribution | Method | 64→128 reconstruction PSNR | 128→256→128 round-trip PSNR |
|---|---|---:|---:|
| Gaussian | Nearest neighbor | 31.21 dB | Effectively exact |
| Gaussian | Default analytic 2× | 30.10 dB | 35.96 dB |
| Gaussian | Mild sharpen 0.1 | 30.21 dB | 36.35 dB |
| Uniform | Nearest neighbor | 35.21 dB | Effectively exact |
| Uniform | Default analytic 2× | 33.94 dB | 39.81 dB |
| Uniform | Mild sharpen 0.1 | 34.07 dB | 40.20 dB |

Nearest-neighbor reconstruction scores better here. Bilinear deliberately trades exact pixel reproduction for smoother enlargement. We do not interpret these numbers as evidence that bilinear improves portrait fidelity. The default was specified before evaluation because its bounded interpolation avoids sharpening overshoot. Holdout mean brightness shift was zero to float32 precision for the default; the sharpening variant shifted it by approximately `6e-8`.

Visual inspection of `figures/comparison.png` confirms smoother pixel transitions, limited sharpening differences, and persistent grid texture. There was no blinded human review, real-image evaluation, identity test, or new repetition measurement in this round. The prior diversity edits are retained in the unchanged parent; this output head does not increase generative diversity. The round5 stateful sampler remains a separate tool for reducing repeats and can select portraits before applying this head.

## Validation

The head's largest absolute error against PyTorch bilinear interpolation was `1.1921e-7`; constants and the impulse pattern matched exactly. The full decoded default and sharpened graphs were compared against the Torch generator plus head on zero, Gaussian, and uniform inputs, using batches 1, 3, and 2. The largest error across both exports and both arithmetic backends was `7.7487e-7`. All output shapes were `[batch,256,256,3]`.

**Native LiteRT acceptance remains unverified.** The actual acceptance command exited 2 with `NOT RUN` because this environment has neither LiteRT nor TensorFlow installed. Decoding and executing a FlatBuffer with our evaluator establishes numerical agreement for our supported operators; it does not establish that a native interpreter/delegate will accept or execute the file. The native acceptance helper is supplied for a compatible environment and checks batches 1/2/3 against decoded outputs.

Hashes:

```
model_upscale_2x.tflite
5d66ad7465a1864e898d4d208dd9c6a8f34be2bffdb4fe845b1c3f95a899196b

model_upscale_2x_sharp.tflite
704e136e7d58ce92c34b80f0253477277713fa8ae8be115cca4317de6fa1f4aa
```

## Reproduction and use

Use the existing local ML interpreter, which has Torch, NumPy, and FlatBuffers:

```sh
python3 round6/sample.py --count 8 --seed 88720 --output /tmp/upscaled_portraits.npz
```

This runs the decoded graph and saves float32 images, latent inputs, and a PNG preview. It refuses to overwrite existing output files. Add `--model round6/candidates/model_upscale_2x_sharp.tflite` to compare the sharp variant. The preview helper also needs Pillow; this workstation supplies it via the system Python fallback.

Native acceptance, once a compatible interpreter is installed:

```sh
python round6/validate_native.py
```

For a native application, retain float32 noise input `[N,512]` and update the output handling to `[N,256,256,3]`, RGB floats in `[0,1]`. The input/output signature aliases are inherited, but the output tensor index changes. Query interpreter tensor details rather than hardcoding the parent's output index or 128×128 buffer size. Native compatibility must pass the acceptance gate before integration.

`upscale.py` contains the complete build and experiment. It refuses to overwrite a completed run. To repeat the entire experiment, copy the workspace to a separate location and remove only that copy's round6 generated outputs, retaining the three round6 scripts plus the native acceptance helper, then run `round6/upscale.py`. The existing source model, Torch parent checkpoint, graph specification, and helper modules from previous rounds are dependencies.

## What this establishes

It is possible to append useful, predictable layers without retraining when their behavior can be specified mathematically. This round implements that case: a lightweight interpolation head inside an existing generator. Improved high-resolution facial detail would require a trained or pretrained restoration stage, and would need separate evaluation for identity changes and invented features. This experiment supplies no evidence that arbitrary new learned layers can be initialized to provide those benefits.
