# Round10: overlapping upsampling diagnosis

This round measured periodic RGB edge profiles and compared 23 analytical interventions, including ordinary smoothing controls. The preserved diagnostic export changes only the final upsampling projection with overlap coefficient 0.35 and phase-balance coefficient 0.

On fresh Gaussian and uniform inputs it reduced normalized edge periodicity by about 36%, but failed the frozen 20% reduction requirement for per-image brightness phase4. It remains experimental. The follow-up uses new seeds and separately tunes continuity and phase balance.

The complete method, failed controls, exact formulas, seed schedule, validation, results, and limitations for both rounds are recorded in [the round11 technical write-up](../round11/TECHNICAL_WRITEUP.md). Round10's original plan, interrupted source snapshot, corrected resume source, per-candidate patches, and logs remain in this directory.
