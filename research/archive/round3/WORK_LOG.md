# Third round structure experiment log

## Step 1 Review evidence and establish the limits

Read the current architecture, prior checksums and prior measurements. No training
data, encoder or input-generation code was found in the relevant workspace files.
Asked whether those sources exist; proceeded with an independent structural
prototype while awaiting an optional answer. Consulted the primary Net2Net,
Network Morphism and StyleGAN papers for function-preserving expansion and
learned controls, and the limited-data GAN paper for a conditional training path.

## Step 2 Derive the transferred architecture

Replace only the contributing first-layer 2×2 weight positions with a dense
projection, flattening output channel then row then column. Repeat each original
channel bias four times. Preserve the asymmetric early 3×3 transposed convolution.
Re-express each later non-overlapping 2×2 transposed convolution as a 1×1 phase
projection followed by pixel shuffle, with a separate shared channel bias.
Re-express the final pointwise transforms as ordinary 1×1 convolutions.

## Step 3 Add a new block without changing the initial function

Insert a two-convolution residual branch at 32×32 with 64 channels. The first
convolution is a center identity, the final convolution and all branch biases are
zero. There is no activation after the skip addition. This creates 73,856 new
parameters; inherited parameters are frozen. No optimization or training occurs.

## Step 4 Checkpoint and validate

Saved the source code, state and manifest in checkpoint 001. Compared to v1 on
zero, Gaussian, uniform and scaled-Gaussian input sets, 273 inputs total, seed
88430. Maximum output error was 1.49e-6, under the predeclared 2e-5 absolute and
relative tolerances. All outputs were finite. A standalone signed feature probe
was unchanged by the new branch, and state serialization round-trip was exact.
The prototype has 7,618,339 parameters, 58.93% fewer than v1's stored floats.

## Step 5 Interpret and document

Wrote RESTRUCTURING.md distinguishing implemented equivalence from proposed
learning, specifying data requirements and the unresolved input prior. The
prototype is a PyTorch state, not a TFLite export, and does not claim improved
image quality, identity coverage or angle control. Prior TFLite models are kept.
