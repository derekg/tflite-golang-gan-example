# Reproducing the portrait experiment

`evaluate.py` runs the two distributed TFLite files through a decoded-graph Torch evaluator. It reproduces the original-versus-round11 measurements without intermediate checkpoints or new data. This evaluator is separate from the browser implementation and is not native LiteRT.

From the repository root:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r research/requirements.txt
python research/evaluate.py --output evaluation-new.json
```

Default: 256 vectors for each of Gaussian (seed 89202), uniform [−1,1] (89203), and legacy uniform [0,1] (89304). The last law was a post-selection audit after the model was frozen. `--count 8` is a quick smoke check and does not reproduce the published 256-sample estimates. Existing output files are never overwritten. Tested with Python 3.11, Torch 2.9, and NumPy 2.4; numerical results can differ slightly across arithmetic libraries.

`metrics.py` contains the unchanged numerical functions extracted from the historical scripts, with source annotations. In particular, `edge_grid` normalizes phase-folded edge variation by mean edge magnitude, `artifacts` computes per-image high-pass brightness phase4 RMS, and `measure` computes binomial-smoothed broad-edge energy. None is a human realism score. `expanded` implements the analytical 3×3 projection formula.

The baseline and new output get exactly the same float32 vectors. Model hashes are checked before inference. The gallery uses independent fixed seeds 88920 and 88921, first eight vectors each, without screening or random flips. `../evidence/gallery.json` describes its edge exposure; `../web/samples.png` and `edges.png` store two 128px columns and sixteen 128px rows.

## Historical record

`archive/` preserves experiment source files, technical notes, and append-only logs through round11. Machine-specific paths were normalized for publication; the untouched originals remain in the working archive. These are historical records: statements such as “browser still on round9” or “intended noise law unknown” describe what was known at that checkpoint. The 2021 article was subsequently found to document MetFaces training, and its Go example samples uniform [0,1]. The new browser app compares original against native round11.

The historical sweep runners depend on intermediate `.pt`/`.tflite` checkpoints not bundled here. They document transformations and selection, but are **not a turnkey reconstruction of every intermediate model**. Use the two distributed final artifacts and `evaluate.py` for independently repeatable inference and metrics. Round10 and round11 development, frozen selections, holdouts, and diagnostics are included under `../evidence/round10/` and `../evidence/round11/`.

The final model branches from round9, changing only its final upsampler projection tensor and adding edge padding. For the four phase projections `Wp`, their mean `M`, center impulse `delta`, and half-pixel bilinear phase kernel `Bp`:

```text
Wprime_p = (Wp - b*(Wp - M))*delta + a*M*(Bp - delta)
selected: a=0.5, b=0.2
```

This preserves the phase-average response for constant spatial input algebraically. Nonlinear output RGB still requires measurement. The optional 256px model from the experiment is interpolation, not learned new detail, and is not used in this comparison.

## Limits

The original model, its early improvements, and every local checkpoint are preserved. The browser CPU/worker and simulated control tests pass. Actual browser rendering, GPU execution, and native LiteRT acceptance still need device testing. The finite coefficient selection used generated feedback; it should not be presented as a no-feedback or one-shot repair. Diversity proxies do not identify people, and cleaner texture does not correct missing anatomy.
