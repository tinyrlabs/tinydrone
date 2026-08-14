/**
 * inference.h — tinycml CNN inference (ESP32)
 *
 * Model: Conv(16)→ReLU→Pool→Conv(32)→ReLU→Pool→FC(64)→FC(4)
 * Input: 32x32x3 NCHW, [-1, 1]
 * Output: 4 sınıf olasılıkları + tahmin
 */

#ifndef TINYDRONE_INFERENCE_H
#define TINYDRONE_INFERENCE_H

#include <stdint.h>

#define TD_N_CLASSES 4

/** Init model (ağırlıklar model.h'ten). İlk çağrıdan önce zorunlu. */
void inference_init(void);

/**
 * 32x32x3 input → sınıf olasılıkları.
 *
 * @param input  3072 double, NCHW, [-1, 1]
 * @param probs  4 double çıkış (softmax)
 * @return argmax sınıf index (0-3)
 */
int inference_run(const double *input, double *probs);

#endif /* TINYDRONE_INFERENCE_H */
