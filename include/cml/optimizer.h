#ifndef CML_OPTIMIZER_H
#define CML_OPTIMIZER_H

#include "cml/vector.h"

/* Callback and state contracts for the batch gradient descent milestone.
 * The optimizer borrows context and parameters. The callbacks write a finite
 * loss or gradient to optimizer-provided storage on success and propagate
 * CMLStatus errors. Gradient output is pre-sized to parameters->size.
 */
typedef CMLStatus (*CMLObjectiveFunction)(const CMLVector *parameters,
                                          void *context, double *out_loss);
typedef CMLStatus (*CMLGradientFunction)(const CMLVector *parameters,
                                         void *context,
                                         CMLVector *out_gradient);

typedef struct CMLOptimizerConfig {
  double learning_rate;      /* finite and positive */
  size_t max_iterations;     /* maximum parameter updates, positive */
  double loss_tolerance;     /* finite and nonnegative; zero disables */
  double gradient_tolerance; /* finite and nonnegative; zero disables */
} CMLOptimizerConfig;

/* Optimizer-owned configuration and scratch state. */
typedef struct CMLOptimizer CMLOptimizer;

typedef enum CMLOptimizationStatus {
  CML_OPTIMIZATION_CONVERGED,
  CML_OPTIMIZATION_MAX_ITERATIONS,
  CML_OPTIMIZATION_DIVERGED,
  CML_OPTIMIZATION_ERROR,
  CML_OPTIMIZATION_CANCELLED,
} CMLOptimizationStatus;

typedef struct CMLOptimizationResult {
  CMLOptimizationStatus status;
  CMLStatus error;   /* CML_SUCCESS for a normal stopping condition */
  size_t iterations; /* completed parameter updates */
  double initial_loss;
  double final_loss;
  double gradient_norm; /* L2 norm at the final parameters */
} CMLOptimizationResult;

#endif /* CML_OPTIMIZER_H */
