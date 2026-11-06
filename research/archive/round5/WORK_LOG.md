# Fifth round log

## User clarification and scope

The user confirmed that the intended model input is random noise and prioritized
cleaner portraits and fewer repeats. The exact noise distribution and scale were
not specified. Use standard-normal noise as the primary assumption and retain
uniform −1 to 1 as a model robustness check. Pose work is not the priority.

## Combine the two useful weight branches

Freeze calibration seed 88600 and development seed 88601, 128 vectors each.
Reserve 512-vector holdouts at seeds 88602 and 88603. Verify that v1 and the earlier
diversity model differ only in the first weight tensor. Transfer the latter into
the compact architecture and recalibrate phase cleanup against each candidate's
own activations. Compare existing cleanup, diversity only, half-strength combined
and full-strength combined variants. Set cleanup, detail, clipping, shape-spread
and repetition guards before looking at development results.

Both combined variants passed the guards. Choose the full variant for its larger
nearest-shape-distance gain; checkpoint selection before opening holdouts. Export
a new 31,540,512-byte TFLite graph and PyTorch state without modifying old files.
Check graph arithmetic against both the module and NumPy, including dynamic batches.

## Evaluate the combined model

On unseen Gaussian/uniform inputs, reflection-invariant nearest shape distance
increased 11.24%/4.17% versus the cleanup-only model. Shape rank increased
12.94%/10.86%, but average shape distance decreased 9.64%/11.34%. Four-pixel grid RMS
rose 6.70%/2.60%, an explicit tradeoff versus the strongest cleanup-only model.
Edge energy increased 1.94%/3.97%. This combines most prior cleanup with lower local
descriptor repetition; it does not prove new identities or superior aesthetics.

## Implement persistent recent-output memory

Add a sampler that generates a pool, screens extreme grid/clipping/low-contrast
cases, and selects samples far from recent and current-batch descriptors.
Descriptors are mean-centered, contrast-normalized 16×16 RGB, matched across
horizontal reflection. The cache stores 128 recent descriptors, bound to model
checksum and noise distribution. Threshold exhaustion is explicitly reported;
no unlimited retry or identity-uniqueness guarantee is implied.

## Reject the first sampler policy and revise it

An initial 95th-percentile grid screen reduced close descriptor matches, but its
selected outputs had 6.45% higher per-image grid RMS on its evaluation sequence.
Preserve the code, calibration, results and preview in checkpoint 006. Do not
recommend it as a cleanup improvement.

Declare a revision with grid quantiles 0.50, 0.75 and 0.95. Use fresh development
request seeds 88612–88615, with identical cached candidate pools across policies.
Require no mean grid increase, at least 25% higher nearest shape distance, retained
coarse contrast and bounded clipping. Select the 0.75 quantile policy and freeze
it in checkpoint 007 before evaluating fresh seeds 88650–88657.

## Evaluate revised selection and check persistence

Across eight new requests of eight outputs each, selected from 128 candidates per
request, nearest shape distance rose from 0.17979 to 0.29137. The fraction with a
neighbor below cosine distance 0.10 fell from 18.75% to zero in this 64-image
sequence. Per-image grid RMS fell from 0.009296 to 0.008317; clipping fell, and
coarse contrast fell about 7.35%. No novelty-threshold fallback was needed.
This selection effect uses 16 candidate inferences per displayed image. It is a
fixed-sequence descriptor comparison, not an identity or human preference test.

Save and reload memory, present the identical pool again, and verify that the
second request chooses different indices and the history grows from 4 to 8.
Document the single-writer cache and explicit descriptor-exhaustion behavior.
Preserve all older models and write technical documentation and checkpoints.

## Final usability and acceptance checks

Changed the sampler default from a fixed demonstration seed to fresh randomness, recording the actual seed so results remain reproducible. Explicit evaluation seeds and all reported measurements are unchanged. Checked the CLI, bounded-history loading, a uniform-noise request, fresh default seeds, syntax, source hashes and export signatures. Native acceptance exited 2 with an explicit NOT RUN message because the runtime is absent. The combined file and descriptor policy remain experimental pending native application acceptance.
