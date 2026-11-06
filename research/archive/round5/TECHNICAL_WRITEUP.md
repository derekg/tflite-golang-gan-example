# Cleaner portraits and fewer recent repeats from random noise

The intended input is now confirmed to be random noise. We can focus on two
separate mechanisms: improving the model's artifact behavior and avoiding similar
outputs in the generation policy. This round combines the earlier diversity edit
with freshly calibrated cleanup, then adds persistent recent-output memory.
All previous models are preserved.

The new model is a balanced alternative to the cleanup-only model. It retains
most of that cleanup and reduces local repetition by coarse shape metrics, at a
small increase in the grid metric. The sampler provides a stronger practical
repetition-control effect, but requires extra candidate generations. Neither
change establishes new identity coverage or human-rated artistic quality.

## The combined weight and structure candidate

The earlier diversity model changes only the first learned weight tensor relative
to v1. We verified that all other stored tensors match, transferred that edit into
the compact architecture, and recomputed the phase correction from the resulting
activations. Reusing the old phase offsets blindly would have assumed the same
feature statistics after the latent rebalancing.

Calibration used 128 Gaussian inputs from seed 88600. Development used another
128 from seed 88601. Four configurations compared the cleanup-only model, the
diversity edit without extra cleanup, and half/full combined strengths. The
predeclared guards limited phase error, loss of edges, clipping and loss of
average shape spread, while requiring at least 5% higher reflection-invariant
nearest shape distance. The full combination was frozen before evaluating the
new holdouts.

The TFLite file retains the 39-operation compact graph, float32 `[batch,512]`
input and `[batch,128,128,3]` output. It is 31,540,512 bytes. The exact random-noise
distribution and scale were not specified, so standard normal is the primary
assumption and uniform −1 to 1 remains a robustness check.

| Change versus the previous cleanup-only model | Gaussian noise | Uniform noise |
| --- | ---: | ---: |
| Reflection-invariant nearest shape distance | +11.24% | +4.17% |
| Shape effective rank | +12.94% | +10.86% |
| Average shape distance | −9.64% | −11.34% |
| Four-pixel phase RMS | +6.70% | +2.60% |
| Edge energy | +1.94% | +3.97% |

Each distribution used 512 unseen inputs, seeds 88602 and 88603. Lower local
repetition and higher rank coexist with lower average separation, so this is not
a universal increase in variety. The previous cleanup-only model remains useful
when its lower grid metric matters more than the repetition tradeoff.

![Matched cleanup-only and combined model outputs](figures/gaussian_comparison.png)

These are the first 16 Gaussian holdout vectors in order, with identical noise
for each left/right pair. The same recurring portrait archetypes remain visible.

The serialized graph was checked against its module and independent NumPy
arithmetic at batch sizes 1, 2 and 3. Additional serialized/module checks used the
first 16 vectors of each model holdout. Native LiteRT loading remains unverified;
the preview and evaluation runner decodes the TFLite graph using Torch arithmetic.
No optimizer, training loop, new portraits or new identity labels were used.

## Repeat control belongs in generation as well as weights

Unchanged random sampling can revisit the same compositions even when the model
has some additional variation. We added a separate `PortraitSampler` that keeps
recent output descriptors across requests.

```text
random-noise candidate pool
  → compact combined TFLite graph
  → artifact screen
  → compare with recent outputs and current selections
  → select portraits
  → save recent descriptors for the next request
```

The TFLite file itself is stateless. Recent-output avoidance runs in the sampler
or equivalent application logic; using only the model file does not activate it.

The descriptor averages RGB into 16×16 cells, subtracts each channel's spatial
mean, flattens to 768 components and normalizes its length. Distance is the minimum
cosine distance to an ordinary or horizontally reflected reference. This reduces
credit for simple flips, global color offsets or contrast changes. It still
measures composition and shape, not a person's identity.

The memory retains the last 128 descriptors and their reflected equivalents,
bound to the model checksum and noise distribution. Changing either resets
history rather than comparing incompatible model outputs. No images are saved
in the history file. A single sampler instance can serve sequential requests,
or the CLI can reopen the cache for each request. The file is single-writer;
concurrent application requests need a lock or centralized sampler instance.

For the first request, selection starts at a descriptor medoid among eligible
candidates. Later selections maximize minimum distance to recent/current output
descriptors. The desired minimum distance is 0.10. If the finite pool cannot meet
it, the policy chooses the best remaining candidate and reports that fact. If all
remaining candidates fail the artifact screen, it explicitly reports a screening
fallback. The limited generator is not guaranteed to supply an unseen identity.

## Screen artifacts without pretending to have an aesthetic judge

Per-image screening measures four-pixel grid residual, extreme clipping and coarse
contrast. Calibration uses 128 model outputs per noise distribution, seeds 88610
and 88611. The clipping limit is the 95th percentile bounded between 0.0001 and
0.01; the minimum coarse contrast is the fifth percentile. These are engineering
guards against some poor outputs, not a face detector or learned quality model.

The first screen allowed grid values up to the 95th percentile. Although its
selected sequence was more varied, its mean per-image grid measurement was 6.45%
higher than ordinary random selection. Farthest-first novelty can favor unusual
or defective images. We preserved that failed policy and its evidence in
checkpoint 006 rather than treating increased novelty as success on both goals.

We then declared three screen strengths, using grid quantiles 50%, 75% and 95%.
Four fresh development pools at seeds 88612–88615 were reused identically across
the policies. Acceptance required no mean grid increase, at least 25% larger
nearest shape distance, retention of at least 85% of coarse contrast, and a
clipping bound. The 75% screen was the most varied accepted candidate and was
frozen in checkpoint 007 before the final request-sequence evaluation.

## Evaluate a sequence of requests

The revised policy was evaluated on eight fresh Gaussian request seeds,
88650–88657. Each request generated 128 candidates and displayed eight portraits,
keeping the previous request's history. The matched random baseline used the
first eight candidates from each of the same pools. This yields two 64-image
sequences.

| Sequence metric | Ordinary random outputs | Screened repeat-aware outputs |
| --- | ---: | ---: |
| Reflection-invariant nearest shape distance | 0.17979 | 0.29137 |
| Fraction with a neighbor below distance 0.10 | 18.75% | 0% |
| Fraction with a neighbor below distance 0.05 | 3.125% | 0% |
| Mean per-image four-pixel grid RMS | 0.009296 | 0.008317 |
| Clipped-pixel fraction | 0.002023 | 0.000077 |
| Coarse contrast | 0.17816 | 0.16506 |

Nearest shape distance rose about 62%, and per-image grid RMS fell about 10.5%.
Coarse contrast fell about 7.35%, a remaining tradeoff. There were no novelty
threshold fallbacks in this sequence. The comparison does not establish that
every displayed face is a different person, nor that every image looks better.
It evaluates the descriptor and artifact measures used by the policy on one
fixed sequence of pools, rather than an independent human preference benchmark.

![Ordinary random outputs from the final sequence](figures/random_batch.png)

![Screened repeat-aware outputs from the final sequence](figures/repeat_aware_batch.png)

The first 16 outputs of each sequence are shown. They come from the same first
two pools but use different candidate indices; they are not paired latent inputs.
Mirroring is not applied to displayed outputs in this round.

This evaluation used **16 candidate inferences per displayed portrait**. The
normal CLI default of 16 outputs from 128 candidates is an 8× candidate workload,
whose effectiveness has not been separately benchmarked here. Pool size is a
cost/choice tradeoff, not a free improvement. On-device latency and memory have
not been measured. Uniform noise has its own calibrated screen and a functional
CLI check, but this final request-sequence benchmark used Gaussian noise only.

## Persistence and integration checks

We saved and reopened the descriptor memory, then supplied the exact same
candidate pool. The second request chose different indices, and the history
grew from four to eight. This checks persistence and exact pool-repeat avoidance;
it does not prove identity-level novelty. The code limits memory length, reports
fallbacks, and does not retry indefinitely.

An application can port the policy without changing the neural model: generate
a bounded pool, average-pool each result, compute normalized descriptors, screen
artifacts, compare against recent descriptors, and retain the latest history.
The provided Python implementation uses the decoded graph backend. Native model
acceptance and application integration still need testing in the actual runtime.

## Reproduction and artifacts

```sh
# From a fresh experiment copy with the exact preserved source artifacts
python round5/combine.py
python round5/evaluate_sampler.py
python round5/refine_sampling.py

# Generate a repeat-aware batch; reuse the same history path on later calls
python round5/portrait_sampler.py \
  --count 8 --pool 128 --seed 88660 \
  --history round5/runtime/recent_history.npz \
  --output round5/figures/portraits.png

# A later request must use fresh noise or it will reuse its candidate pool
python round5/portrait_sampler.py \
  --count 8 --pool 128 --seed 88661 \
  --history round5/runtime/recent_history.npz \
  --output round5/figures/portraits_next.png

# Native acceptance, when a LiteRT or TensorFlow runtime is available
python round5/validate_native.py
```

The model is `candidates/model_clean_diverse.tflite`, with companion PyTorch state.
The screen is `evaluation/artifact_screen.json`, and the sampler's default input
is Gaussian noise. `--distribution uniform` selects uniform −1 to 1 and its
calibrated screen. A different model needs its own screen calibration; the code
rejects a screen/model checksum mismatch. The example seeds are reproducibility
controls and should vary between real generation requests. Omitting `--seed`
uses fresh randomness and records the chosen seed in the output report.

The combined model SHA-256 is:

```text
a19b85ceffb1eb0f9f8c515cf5e6cfbd0acf5eda861086d9e7de393594de13c3
```

The plan, candidate patches, frozen selections, timestamped events, numerical
results and failed first sampler are retained under this round's checkpoints,
evaluation and logs. The earlier cleanup-only model is preserved as a lower-grid
alternative. A single new file is not presented as a universal improvement:
cleaner output and fewer perceived repeats remain separate objectives, with
explicit selection costs and measurable tradeoffs.
