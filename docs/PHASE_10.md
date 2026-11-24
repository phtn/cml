# Phase 10 — Optimizers and advanced training

Compare parameter updates on the same objective and initialization. SGD uses
the gradient directly. Classical momentum accumulates velocity. AdaGrad
accumulates squared gradients, RMSProp tracks their exponential moving average,
and Adam combines that average with a bias-corrected first moment.

## Build sequence

1. Add an optimizer choice while keeping the existing SGD defaults. Initialize
   fresh moment state for every training run.
2. Keep the square root of each second moment and update it with `hypot`.
   Apply Adam's bias correction at each parameter update.
3. Reuse the batch iterator, reproducible shuffling, learning-rate schedules,
   gradient clipping, and validation stopping from Phase 6.
4. Collect epoch records at the chosen frequency. Pass a candidate to the
   checkpoint callback when monitored loss reaches a new minimum, starting
   with the initial parameters. The callback can write a model using Phase 11.
5. Verify a known first update for every algorithm and compare their training
   on the same XOR network.

Run `make test-advanced-optimizers` to verify the updates and training hooks.
