# Fourth round experiment log

## Step 1 Continue from the preserved compact checkpoint

The user authorized proceeding with structural work. Read the previous code,
architecture, documentation and preservation hashes. No real images, original
input-generation code or encoder are present in the relevant workspace files.
Continue with no optimizer or training. Preserve all earlier TFLite models and
the round-three compact checkpoint. Record timestamped events in logs/events.jsonl.

## Step 2 Freeze calibration and selection rules

Recorded evaluation/plan.json before running candidates. Calibration uses seed
88500 with 128 Gaussian inputs. Development uses independent seed 88501 with 128
inputs. Reserve seeds 88502 and 88503 for 512 Gaussian and 512 uniform held-out
inputs. Declare minimum four-pixel artifact reduction, edge and shape retention,
paired image difference, mean-shift and clipping guards.

## Step 3 Test hand-programmed structural additions

Program the existing 32×32 residual branch with small channel-diagonal neighbor
filters at strengths 0.025, 0.05 and 0.1. Separately calibrate phase-dependent
feature offsets at 32×32 and 64×64: subtract each phase's mean high-pass residual,
with zero mean across phases. Try offset strengths 0.25, 0.5 and 0.75, alone and
combined, plus one smoothing/offset combination. These are generated-activation
calibrations, not training on new portraits. Save every candidate patch, checksum,
metrics and sequential preview in checkpoints 001 through 011.

## Step 4 Review and freeze the candidate

Smoothing alone reduced the four-pixel metric by at most about 3%, with detail
loss, and failed the declared minimum artifact reduction. A 64×64 phase-offset
layer at strength 0.75 reduced the development metric about 48% with roughly 2%
lower edge energy. Adding a second phase layer gave negligible extra benefit.
After visual review, select the single offset layer. Save selection.json in
checkpoint 012 before opening the held-out samples. Leave the general 32×32
spatial branch inactive; its usefulness was not established by this experiment.

## Step 5 Build a genuinely compact TFLite file

Write an explicit FlatBuffer builder using the official TFLite schema. Rebuild
all buffers so unused first-layer weights are actually absent. Transfer the
dense stem in NHWC order, preserve the early asymmetric crop, convert later
upsamplers to Conv2D and DepthToSpace, retain the residual branch, and add the
selected periodic-offset tensor after the 64×64 activation. Preserve signature
aliases, input/output dimensions and a dynamic batch dimension. Export a compact
identity model and the refined model as separate files.

TensorFlow NHWC DepthToSpace and PyTorch PixelShuffle use different channel phase
orders. Check the official operation description and explicitly reorder weights
during export. An independent graph runner reads the actual serialized graph,
rather than assuming the original module topology. Implement Torch and NumPy
arithmetic paths and compare both to the module at batches 1, 2 and 3.

## Step 6 Evaluate the frozen export

On 512 unseen Gaussian inputs, four- and eight-pixel phase RMS fell 52.35% and
50.39%. On 512 unseen uniform inputs, they fell 36.53% and 34.70%. Edge energy fell
1.99% and 0.90%; reflection-invariant nearest shape distance changed +0.15% and
+0.26%, effectively unchanged for interpretation. RGB RMS changes were 0.00772
and 0.00906. Check the serialized graph on all 1,024 held-out vectors: maximum
module disagreement was 1.19e-7. Do not claim new identities or pose controls.

The refined file is 31,540,512 bytes, 57.50% smaller than v1. Its 39-operation
graph includes the new structure. No native runtime is installed. The separate
native acceptance script exited 2 with an explicit NOT RUN message; this is
recorded in logs, not treated as a passing native test.

## Step 7 Document the result and the next data requirement

Produce a new technical article and record the data requirements in
DATA_REQUIREMENTS.json. Keep earlier publication artifacts unchanged. The new
file is experimental until native LiteRT acceptance succeeds. No actual dataset
was cleaned or enlarged, and no training, new identity coverage or controllable
3D yaw is claimed. Those steps require the real input protocol and suitable data.

## Step 8 Final acceptance and preservation audit

Ran the documented graph-validation CLI with the selected patch and the compact-preview CLI on two inputs. Python syntax checks passed. Verified all six earlier protected model/checkpoint hashes, 12 new patch hashes, output and input signatures, operation references and 16-byte constant-buffer alignment in both exported files. The model source, serialized graph and independent NumPy arithmetic agree within recorded tolerances; a native runtime test was explicitly NOT RUN. Reviewed Gaussian and uniform paired comparisons. Packaged a self-contained HTML article, editable Markdown, code and evidence bundle, and a separate delivery checkpoint. No prior model or publication artifact was overwritten.
