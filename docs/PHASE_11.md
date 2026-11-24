# Phase 11 — Evaluation and model persistence

Separate training, validation, and test data. Fit preprocessing on training
data, use validation to choose a model or decision threshold, and calculate
final metrics on held-out test data.

## Build sequence

1. Reuse MAE, MSE, and RMSE. Add the coefficient of determination and report
   confusion matrices, per-class rates, and macro, weighted, and micro averages.
2. Split by class when class frequencies matter. Reuse a deterministic
   permutation across k-fold splits and expose batch indices without copying rows.
3. Use model dimension and parameter queries to inspect a fitted model. Keep
   training history separate from its inference parameters.
4. Write a versioned binary format containing the model kind, dimensions,
   topology or classification settings, and double-precision parameters.
   Encode fixed-width values little-endian and reject invalid dimensions,
   unknown versions, non-finite parameters, and incomplete records.
5. Train a model, save or checkpoint it, reload it in an independent object,
   and compare predictions on held-out inputs.

Run `make test-models example-model-workflow` to verify the workflow.
