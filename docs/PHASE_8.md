# Phase 8 — Neural network architecture

A dense layer computes `z = Wx + b`, then applies an activation. Build a
network by chaining dense layers whose output widths determine the next input
width. Start with one hidden layer and compare a small forward pass to hand
calculations.

## Build sequence

1. Describe layers with their output width and activation. The network copies
   the topology and owns a single contiguous parameter vector.
2. Store each layer's row-major weights, with one row per output unit, followed
   by its biases. Initialize biases to zero. Use seeded uniform initialization
   with He variance for ReLU and Xavier variance for the other activations.
3. Implement identity, ReLU, sigmoid, tanh, and softmax. Subtract the maximum
   logit before exponentiating a softmax row.
4. Run the same forward pass for single samples and batches. Reuse activation
   storage across rows. Return dimension and numerical errors before publishing
   incomplete predictions.

## Completion

Run `make test-network`. The tests compare an explicitly configured two-layer
network against hand-computed outputs and check stable probabilities,
repeatable initialization, and feature shape validation.
