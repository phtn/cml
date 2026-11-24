# Phase 9 — Backpropagation

Use the chain rule to propagate a loss derivative from the output layer to the
inputs. Cache logits and activations from the forward pass for each sample.

## Build sequence

1. Use MSE for arbitrary output activations, binary cross-entropy for sigmoid,
   and categorical cross-entropy for softmax. Compute cross-entropy from logits.
2. For each layer, accumulate the outer product of its logit derivative and
   its input into the weight gradient. Sum logit derivatives into the bias
   gradient. Pass the transposed weight product back to the preceding layer.
3. Multiply by activation derivatives before each weight update. For softmax,
   apply the Jacobian-vector product `p * (g - dot(p, g))`.
4. Average across the batch and pass the gradient to the optimizer. Validate
   it with the central-difference checker on initialized, nonzero parameters.

## Completion

Run `make test-backpropagation example-xor`. The tests verify both hidden and
output layer gradients and show that a reproducibly initialized network learns
all four XOR cases.
