# A portrait GAN, revisited

The original 2021 Go + TensorFlow Lite example now has a browser comparison with a separately edited model. Both models receive the **same 512-value noise vector**. The original file and Go example are preserved.

[Read the technical story](https://derekg.github.io/weight-space.html) · [Open the comparison](https://derekg.github.io/portrait-lab/) · [Original 2021 article](https://derekg.github.io/tflite.html)

## Run the browser app

```sh
git clone https://github.com/derekg/tflite-golang-gan-example.git
cd tflite-golang-gan-example
python3 -m http.server 8000
```

Open **http://localhost:8000/web/**. Serve the repository root so the `model/` directory is available. No JavaScript package installation or inference server is needed. All scripts, saved comparisons, and model weights are included locally.

The saved gallery loads immediately. Live mode downloads about **106 MB**, evaluates both models on each intermediate noise vector, and walks continuously between random endpoints. It displays each pair together, never crossfading finished images. Choose Gaussian, legacy uniform [0,1], or uniform [−1,1] inputs. Pause, change the pace, or reveal edge maps at a shared exposure. Reduced-motion preferences pause the walk initially.

Both outputs are natively **128 × 128** and enlarged equally. The hosted blog copy loads static model files from this repository; the cloned app uses local files. The custom FlatBuffer evaluator uses WebGL2 if its startup numerical checks pass, with a JavaScript CPU fallback. The fallback can take seconds per paired frame.

## What changed

Inspired by [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/), the experiment inspected the graph, removed unreachable storage, rebalanced selected weight/phase responses, and added analytically initialized spatial overlap. No optimizer, gradients, conventional training loop, or new image dataset was used. Coefficients were chosen through finite development sweeps, then evaluated on fresh noise.

The final round11 projection uses shared-field overlap `a=0.5` and phase attenuation `b=0.2` at the final 64→128 upsampler. On 256 Gaussian and 256 uniform [−1,1] inputs, cumulative changes versus the original were:

| Measurement | Gaussian | Uniform [−1,1] |
| --- | ---: | ---: |
| Normalized edge-grid score | −32.7% | −26.7% |
| Per-image brightness phase4 RMS | −73.6% | −69.0% |
| Broad-edge energy | −0.5% | +3.6% |

The last round11 step alone, relative to round9, reduced edge-grid scores about 30% and brightness phase4 about 23%, with roughly 4% less broad-edge energy. Some detail is softer; malformed portraits remain malformed. These are artifact proxies, not percentages of aesthetic improvement or evidence of new identities.

| Model | Bytes | SHA-256 |
| --- | ---: | --- |
| `model/model.tflite` (untouched original) | 74,217,344 | `0074d74ac51f7a67361c2f881faeeebb591f7297c3878f972538b4c1cb95e79b` |
| `model/model-round11.tflite` (native 128px) | 32,068,288 | `306f8796778eade5935903f74a901844ad385af0f445f052625ce31f2ebfeb81` |

## Verification and reproduction

```sh
node web/verify.mjs
node web/verify-controls.mjs
```

The CPU evaluator agrees with independent Python outputs on five paired inputs to within `7.75e-7` on the 0–1 scale. The paired worker and 19 simulated UI checks pass. Actual browser rendering, WebGL behavior, and native TensorFlow/LiteRT acceptance remain unverified in the build environment. The WebGL startup comparison is a runtime check, not a substitute for that acceptance testing.

[research/](research/) includes a runnable frozen-model evaluation, the exact metric functions, and historical experiment sources/notes. [evidence/](evidence/) includes selection plans, development results, holdouts, uncertainty estimates, checksums, and verification records. Historical rounds reference intermediate model checkpoints that are preserved in the working archive but are not all distributed in this repository.

## Original Go example

The 2021 implementation and its bundled native libraries are retained. Its input sampling is uniform [0,1], and it randomly flips the displayed portrait. Those behaviors differ from the aligned Gaussian comparison used in much of the new experiment. The historical native build has not been retested on current platforms.

The [original article](https://derekg.github.io/tflite.html) covers that implementation. [README-2021.md](README-2021.md) preserves the original build notes verbatim, including their historical path example.
