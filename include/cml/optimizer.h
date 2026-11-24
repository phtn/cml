#ifndef CML_OPTIMIZER_H
#define CML_OPTIMIZER_H

#include "cml/vector.h"

#include <stdint.h>

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

/* An optimizer owns a copy of its configuration. Each run starts with fresh
 * state. The input parameter vector is updated only on a successful run.
 * A finite run that reaches the update limit is successful.
 */
CMLOptimizer *cml_optimizer_create(const CMLOptimizerConfig *config);
void cml_optimizer_destroy(CMLOptimizer *optimizer);
CMLStatus cml_optimizer_minimize(const CMLOptimizer *optimizer,
                                 CMLVector *parameters,
                                 CMLObjectiveFunction objective,
                                 CMLGradientFunction gradient, void *context,
                                 CMLOptimizationResult *result);

/* A batch callback computes the mean loss, gradient, or both. If indices is
 * NULL, use the first count samples. A NULL output is skipped. The callback
 * must fill every gradient element when an output is provided.
 */
typedef CMLStatus (*CMLBatchFunction)(const CMLVector *parameters,
                                      const size_t *indices, size_t count,
                                      void *context, double *loss,
                                      CMLVector *gradient);

typedef enum CMLLearningRateSchedule {
  CML_RATE_CONSTANT,
  CML_RATE_INVERSE_TIME,
  CML_RATE_STEP,
  CML_RATE_EXPONENTIAL,
} CMLLearningRateSchedule;

typedef struct CMLTrainingRecord {
  size_t epoch;
  size_t updates;
  double training_loss;
  double validation_loss; /* NaN when validation is absent */
  double gradient_norm;
  double parameter_norm;
  double learning_rate;
  bool best; /* a new best validation model */
} CMLTrainingRecord;

/* The callback borrows the record and candidate parameters until it returns.
 * Return false to cancel and preserve the parameters supplied to training.
 */
typedef bool (*CMLTrainingObserver)(const CMLTrainingRecord *record,
                                    const CMLVector *parameters, void *context);

typedef enum CMLOptimizerMethod {
  CML_OPTIMIZER_SGD,
  CML_OPTIMIZER_ADAGRAD,
  CML_OPTIMIZER_RMSPROP,
  CML_OPTIMIZER_ADAM,
} CMLOptimizerMethod;

typedef struct CMLTrainingHistory {
  CMLTrainingRecord *records;
  size_t size;
  size_t capacity;
} CMLTrainingHistory;

/* Release storage for a zero-initialized history and reset it to empty. */
void cml_training_history_destroy(CMLTrainingHistory *history);

/* Save or inspect a candidate. A checkpoint error stops the run and its
 * status is returned. With no validation set, training loss selects
 * checkpoints.
 */
typedef CMLStatus (*CMLTrainingCheckpoint)(const CMLTrainingRecord *record,
                                           const CMLVector *parameters,
                                           void *context);

typedef struct CMLTrainingOptions {
  CMLOptimizerConfig optimizer;
  size_t epochs;
  size_t batch_size; /* zero means the full training set */
  bool shuffle;
  uint64_t seed;
  double momentum;  /* classical momentum, in [0,1) */
  double clip_norm; /* zero disables norm clipping */
  CMLLearningRateSchedule schedule;
  double decay;      /* nonnegative; step decay uses a factor in (0,1] */
  size_t decay_step; /* epochs per step for CML_RATE_STEP */
  size_t min_epochs;
  size_t patience; /* consecutive epochs without improvement, zero disables */
  double min_improvement;
  double parameter_tolerance; /* zero disables */
  size_t record_every;        /* zero is equivalent to one */
  CMLTrainingObserver observer;
  void *observer_context;
  CMLOptimizerMethod method;
  double beta1, beta2, epsilon; /* adaptive optimizers */
  CMLTrainingHistory *history;  /* optional, zero-initialized; records append */
  CMLTrainingCheckpoint checkpoint;
  void *checkpoint_context;
} CMLTrainingOptions;

typedef struct CMLTrainingResult {
  CMLOptimizationResult optimization;
  size_t epochs;
  size_t best_epoch;      /* zero means the initial parameters */
  double validation_loss; /* NaN when validation is absent */
} CMLTrainingResult;

CMLTrainingOptions cml_training_options_default(void);

/* Training and validation use the same callback with separate contexts.
 * When validation_count is positive, restore the best validation parameters
 * (including the initial candidate) at any successful stop.
 */
CMLStatus cml_train(CMLVector *parameters, CMLBatchFunction batch,
                    size_t training_count, void *training_context,
                    size_t validation_count, void *validation_context,
                    const CMLTrainingOptions *options,
                    CMLTrainingResult *result);

/* Relative error is |analytic - numeric| / max(|analytic|, |numeric|).
 * Each component passes if either tolerance is met.
 */
typedef struct CMLGradientCheckResult {
  size_t worst_index;
  double max_absolute_error;
  double max_relative_error;
  bool passed;
} CMLGradientCheckResult;

CMLStatus cml_gradient_check(const CMLVector *parameters,
                             CMLObjectiveFunction objective,
                             CMLGradientFunction gradient, void *context,
                             double step, double abs_tolerance,
                             double rel_tolerance, CMLGradientCheckResult *out);

#endif /* CML_OPTIMIZER_H */
