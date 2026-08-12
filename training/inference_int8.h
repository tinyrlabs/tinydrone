/**
 * inference_int8.h — int8 CNN inference API
 */

#ifndef TINYDRONE_INFERENCE_INT8_H
#define TINYDRONE_INFERENCE_INT8_H

/** Init int8 model (scale hesaplama). */
void inference_int8_init(void);

/**
 * 32x32x3 float input (NCHW, [-1,1]) → int8 inference.
 * @return argmax sınıf (0-3), probs dolu döner.
 */
int inference_int8_run(const double *input, double *probs);

#endif /* TINYDRONE_INFERENCE_INT8_H */
