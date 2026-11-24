#ifndef CML_METRICS_H
#define CML_METRICS_H

#include "cml/vector.h"

typedef struct CMLConfusionMatrix {
  size_t true_negative, false_positive, false_negative, true_positive;
} CMLConfusionMatrix;

typedef struct CMLBinaryMetrics {
  CMLConfusionMatrix confusion;
  double accuracy, precision, recall, f1, specificity, balanced_accuracy;
} CMLBinaryMetrics;

/* Rates with a denominator of zero return zero. */
CMLStatus cml_binary_metrics(const CMLVector *actual,
                             const CMLVector *predicted, CMLBinaryMetrics *out);

typedef struct CMLRocPoint {
  double threshold, false_positive_rate, true_positive_rate;
} CMLRocPoint;

typedef struct CMLRocCurve {
  CMLRocPoint *points;
  size_t size;
  double auc;
  double best_f1;
  double best_threshold;
} CMLRocCurve;

/* Scores are probabilities in [0,1]. Both classes must occur. Tied scores
 * enter the same ROC step. Among F1 ties, the largest threshold wins.
 * The caller owns the returned curve.
 */
CMLStatus cml_roc_curve(const CMLVector *actual, const CMLVector *scores,
                        CMLRocCurve **out);
void cml_roc_curve_destroy(CMLRocCurve *curve);

#endif
