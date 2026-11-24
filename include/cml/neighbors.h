#ifndef CML_NEIGHBORS_H
#define CML_NEIGHBORS_H

#include "cml/dataset.h"

typedef struct CMLKNN CMLKNN;

/* Labels are finite doubles and each distinct value is a separate class.
 * Fitting copies the training data. Distance ties use training row order;
 * vote ties use the smallest numeric label.
 */
CMLKNN *cml_knn_create(size_t neighbors);
void cml_knn_destroy(CMLKNN *model);
CMLStatus cml_knn_fit(CMLKNN *model, const CMLDataset *data);
CMLStatus cml_knn_predict(const CMLKNN *model, const CMLVector *input,
                          double *out);

typedef struct CMLKMeans CMLKMeans;
typedef struct CMLKMeansResult {
  size_t iterations;
  double inertia; /* total squared distance to the closest center */
  bool converged;
} CMLKMeansResult;

CMLKMeans *cml_kmeans_create(size_t clusters);
void cml_kmeans_destroy(CMLKMeans *model);

/* Use a seeded first sample and farthest-first remaining centers. Empty
 * clusters retain their previous centers. Ties use the lowest cluster index.
 * A failed fit preserves previously fitted centers.
 */
CMLStatus cml_kmeans_fit(CMLKMeans *model, const CMLMatrix *data, uint64_t seed,
                         size_t iterations, double tolerance,
                         CMLKMeansResult *out);
CMLStatus cml_kmeans_get_centers(const CMLKMeans *model, CMLMatrix *out);
CMLStatus cml_kmeans_predict(const CMLKMeans *model, const CMLVector *input,
                             size_t *out);

#endif
