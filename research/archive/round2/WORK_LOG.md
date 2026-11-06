# Second round experiment log

This log begins at the second round. The first round is reconstructed from its saved scripts, metrics and images; it was not logged prospectively.

## Step 0 Preserve the existing model

Copied the improved model, all existing Python scripts, README and first-round results to checkpoints/000_preserved_v1. Recorded SHA-256 hashes of both source models. New experiments use separate files.

## Step 1 Declare hypotheses and selection data

All candidates branch from v1. Development inputs use seed 88421, 128 standard-normal vectors. Bias changes test whether thresholding restricts mixtures. Spectral edits test underrepresented latent directions. Coarse and kernel reflections test geometry; they must be compared to a true pixel reflection before calling them mirrors. Repetition is also measured after matching reflected images, so mirroring cannot earn false novelty credit. No optimizer, gradients or external image data are used.

### Logging correction

The first development run saved the baseline checkpoint, then stopped because the event logger used the same argument name as candidate metadata. Renamed the logger argument and resumed from the existing checkpoint. The failed source version is retained in source/run_experiments.frozen.py.

## Step 1 Declare hypotheses and selection data

All candidates branch from v1. Development inputs use seed 88421, 128 standard-normal vectors. Bias changes test whether thresholding restricts mixtures. Spectral edits test underrepresented latent directions. Coarse and kernel reflections test geometry; they must be compared to a true pixel reflection before calling them mirrors. Repetition is also measured after matching reflected images, so mirroring cannot earn false novelty credit. No optimizer, gradients or external image data are used.

## Step 2 Checkpoint candidate edits and traversals

Saved every candidate as a compressed patch with source hash, patch hash, hypothesis, full development metrics and the first 16 images in seed order. Saved leading latent principal-direction traversals at coefficients −3 through +3. Results are in evaluation/development_metrics.json. Selection and visual review happen after this checkpoint; no holdout images have been evaluated yet.

## Step 3 Review and freeze selection

Additional alpha 0.2 and gain 1.2 improves nearest shape distance and shape rank in development with modest detail change. Stronger flattening gives larger rank but lower pairwise separation; reject it. Bias edits give negligible structural gains. Reject coarse and approximate mirror weight edits for visible blur or displacement. The eight principal-direction traversals mainly trade between recurring compositions, illumination and color; no clean yaw-only direction was identified. Select checkpoint 008 before opening the two holdout sets, seeds 88422 and 88423.

## Step 4 Evaluate the frozen models

Evaluated 512 matched standard-normal and 512 matched uniform inputs. Saved full metrics, eight paired blocks per distribution, reference comparisons and sequential image triplets. The explicit mirror model adds REVERSE_V2 on NHWC axis 2; it is a graph-level transformation with zero weight changes and does not count as identity diversity. Native LiteRT execution remains unverified.

## Step 5 Interpret heldout results

Relative to v1, v2 raised reflection-invariant nearest shape distance 12.0% on the 512 Gaussian samples and 6.3% on the 512 uniform samples. The fraction with a close neighbor at cosine distance below 0.05 fell from 14.06% to 7.42% and 14.45% to 10.16%, respectively. Average shape separation decreased 9.1% and 11.3%; gains therefore reflect reduced local repetition rather than a universally wider distribution. All eight Gaussian blocks improved, but only four of eight uniform blocks improved. Retain v2 as an optional experimental variant; do not replace v1.

## Step 6 Add optional batch selection and facing direction control

Added sample_diverse.py, a separate inference policy that generates 256 samples and greedily chooses 16 using reflection-invariant shape distance, initialized at a medoid. Optional balanced mirroring flips half the displayed images. This changes selection and presentation, not weights, identities or 3D pose. Seed 88424 is a demonstration rather than an unbiased generalization benchmark. Exact selected indices and mirror flags are saved.

## Step 7 Package the technical writeup and audit

Wrote the full repository technical article in Markdown and self-contained HTML, with formulas, first-round reconstruction, every second-round hypothesis, rejection reasons, held-out results and limitations. The renderer initially expected the wrong figure/table counts; corrected its validation to four of each. Python syntax checks passed. Verified 15 candidate patch hashes, frozen model equality, exported graph counts and input/output shapes, and both protected source hashes. Reviewed all comparison image sets visually. Browser preview was unavailable, and a local HTTP server could not bind under the sandbox; HTML structure and embedded image assets were checked programmatically, but the full HTML layout has not been reviewed in a browser. Native LiteRT execution is still outstanding. Final files are listed in evaluation/final_audit.json. The publication ZIP includes code, figures, logs and measurements, but excludes models and large numerical patches. No external post or upload was made.
