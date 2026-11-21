
/* ==============================================================================
 * src/math.c — CML Mathematical Utilities Implementation
 * ------------------------------------------------------------------------------
 */

#include "cml/math.h"

#include <math.h>

bool cml_is_close(double a, double b, double tol) {
  if (isnan(a) || isnan(b)) {
    return false;
  }
  if (isinf(a) || isinf(b)) {
    return a == b;
  }
  double diff = fabs(a - b);
  if (diff <= tol) {
    return true;
  }
  double max_mag = fmax(fabs(a), fabs(b));
  return diff <= (tol * max_mag);
}

double cml_clamp(double x, double min_val, double max_val) {
  if (x < min_val)
    return min_val;
  if (x > max_val)
    return max_val;
  return x;
}

double cml_max(double a, double b) { return fmax(a, b); }

double cml_min(double a, double b) { return fmin(a, b); }

double cml_sign(double x) {
  if (x > 0.0)
    return 1.0;
  if (x < 0.0)
    return -1.0;
  return 0.0;
}
