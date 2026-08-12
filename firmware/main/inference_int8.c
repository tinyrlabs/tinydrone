/**
 * inference_int8.c — int8 quantize edilmiş CNN inference
 *
 * Model: tinydrone_model_int8.h (ağırlıklar) + model.h (float bias)
 * Tüm matris çarpımları int8×int8→int32 accumulate → scale çarpımı.
 * ESP32 için: ~4x daha az bellek, 2-4x daha hızlı.
 *
 * Bu dosya hem host testi hem firmware için kullanılabilir (ESP-IDF'den bağımsız).
 */

#include "inference_int8.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#include "tinydrone_model_int8.h"
#include "tinydrone_model.h"  /* float bias değerleri için */

#define IMG_H 32
#define IMG_W 32
#define IMG_C 3

/* Giriş normalizasyonu: [-1, 1] → int8 scale */
#define INPUT_SCALE (2.0 / 255.0)

/* Aktivasyon scale'leri (kalibrasyon aralıklarından) */
static float a_scale_conv1, a_scale_conv2, a_scale_fc1;
static float a_zero_conv1, a_zero_conv2, a_zero_fc1;

/* Doğrudan convolution (im2col'suz — int8 için daha verimli) */
static void conv2d_i8(const int8_t *input, int in_c, int in_h, int in_w,
                      const int8_t *w, const float *w_scale, int out_c,
                      int k, int pad,
                      const double *bias, float input_scale, int in_zero,
                      int8_t *output, int *out_h, int *out_w,
                      float out_scale, float out_zero) {
    int oh = in_h + 2 * pad - k + 1;
    int ow = in_w + 2 * pad - k + 1;
    *out_h = oh;
    *out_w = ow;

    for (int oc = 0; oc < out_c; oc++) {
        const int8_t *wrow = &w[oc * in_c * k * k];
        float ws = w_scale[oc];
        for (int oy = 0; oy < oh; oy++) {
            for (int ox = 0; ox < ow; ox++) {
                int32_t acc = 0;
                for (int ic = 0; ic < in_c; ic++) {
                    for (int ky = 0; ky < k; ky++) {
                        int iy = oy + ky - pad;
                        if (iy < 0 || iy >= in_h) continue;
                        for (int kx = 0; kx < k; kx++) {
                            int ix = ox + kx - pad;
                            if (ix < 0 || ix >= in_w) continue;
                            /* (x_q - x_zero) * w_q — zero-point düzeltmesi */
                            int32_t iv = (int32_t)input[ic * in_h * in_w + iy * in_w + ix] - (int32_t)in_zero;
                            int32_t wv = (int32_t)wrow[ic * k * k + ky * k + kx];
                            acc += iv * wv;
                        }
                    }
                }
                /* float bias ekle: out = acc * (ws * input_scale) + bias */
                float val = (float)acc * ws * input_scale + (float)bias[oc];
                /* scale → int8 */
                int32_t q = (int32_t)lroundf(val / out_scale + out_zero);
                if (q > 127) q = 127;
                if (q < -128) q = -128;
                output[oc * oh * ow + oy * ow + ox] = (int8_t)q;
            }
        }
    }
}

/* int8 maxpool */
static void maxpool_i8(const int8_t *input, int c, int h, int w,
                       int8_t *output, int *oh_out, int *ow_out) {
    int oh = h / 2, ow = w / 2;
    *oh_out = oh;
    *ow_out = ow;
    for (int ic = 0; ic < c; ic++) {
        for (int oy = 0; oy < oh; oy++) {
            for (int ox = 0; ox < ow; ox++) {
                int8_t m = input[ic * h * w + (2 * oy) * w + (2 * ox)];
                int8_t v = input[ic * h * w + (2 * oy) * w + (2 * ox + 1)];
                if (v > m) m = v;
                v = input[ic * h * w + (2 * oy + 1) * w + (2 * ox)];
                if (v > m) m = v;
                v = input[ic * h * w + (2 * oy + 1) * w + (2 * ox + 1)];
                if (v > m) m = v;
                output[ic * oh * ow + oy * ow + ox] = m;
            }
        }
    }
}

static void relu_i8(int8_t *data, int n, int8_t zero) {
    for (int i = 0; i < n; i++)
        if (data[i] < zero) data[i] = zero;
}

/* FC: int8 GEMM → int32 → float çıkış (input-major düzen: w[i*out_dim+oc]) */
static void fc_i8(const int8_t *input, int in_dim,
                  const int8_t *w, const float *w_scale, int out_dim,
                  const double *bias, float input_scale,
                  float *output) {
    for (int oc = 0; oc < out_dim; oc++) {
        int32_t acc = 0;
        for (int i = 0; i < in_dim; i++)
            acc += (int32_t)input[i] * (int32_t)w[i * out_dim + oc];
        output[oc] = (float)acc * (w_scale[oc] * input_scale) + (float)bias[oc];
    }
}

/* ReLU sonrası FC (input zaten int8 aktivasyon) — input-major düzen */
static void fc_i8_act(const int8_t *input, int in_dim,
                      const int8_t *w, const float *w_scale, int out_dim,
                      const double *bias, float act_scale,
                      int8_t *output, int8_t out_zero, float out_scale) {
    for (int oc = 0; oc < out_dim; oc++) {
        int32_t acc = 0;
        for (int i = 0; i < in_dim; i++)
            acc += (int32_t)input[i] * (int32_t)w[i * out_dim + oc];
        float val = (float)acc * (w_scale[oc] * act_scale) + (float)bias[oc];
        int32_t q = (int32_t)lroundf(val / out_scale + out_zero);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        output[oc] = (int8_t)q;
    }
}

void inference_int8_init(void) {
    /* Simetrik aktivasyon quantization: scale = max(|min|,|max|)/127, zero=0
     * (asimetrik zero-point int8'de taşar: [-4, 3.4] → zero≈137 > 127) */
    float m1 = fmaxf(fabsf(act_conv1_min), fabsf(act_conv1_max));
    float m2 = fmaxf(fabsf(act_conv2_min), fabsf(act_conv2_max));
    float m3 = fmaxf(fabsf(act_fc1_min), fabsf(act_fc1_max));
    a_scale_conv1 = m1 / 127.0f;
    a_zero_conv1 = 0.0f;
    a_scale_conv2 = m2 / 127.0f;
    a_zero_conv2 = 0.0f;
    a_scale_fc1 = m3 / 127.0f;
    a_zero_fc1 = 0.0f;
}

int inference_int8_run(const double *input_fp, double *probs) {
    /* Girişi int8'e çevir: [-1,1] → scale 2/255, zero 128 (asimetrik) */
    /* Asimetrik kuantizasyon: input range [-1, 1] → scale = 2/255, zero = 0 (simetrik) */
    int8_t X[IMG_C * IMG_H * IMG_W];
    for (int i = 0; i < IMG_C * IMG_H * IMG_W; i++) {
        float v = (float)input_fp[i] / INPUT_SCALE;
        int32_t q = (int32_t)lroundf(v);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        X[i] = (int8_t)q;
    }

    /* Conv1 → ReLU → Pool1 */
    int8_t c1o[16 * 32 * 32];
    int h1, w1;
    conv2d_i8(X, 3, 32, 32, c1_w_i8, c1_w_scale, 16, 3, 1, c1_b, INPUT_SCALE, 0,
              c1o, &h1, &w1, a_scale_conv1, a_zero_conv1);
    relu_i8(c1o, 16 * h1 * w1, (int8_t)a_zero_conv1);
    int8_t p1o[16 * 16 * 16];
    int ph1, pw1;
    maxpool_i8(c1o, 16, h1, w1, p1o, &ph1, &pw1);

    /* Conv2 → ReLU → Pool2 */
    int8_t c2o[32 * 16 * 16];
    int h2, w2;
    conv2d_i8(p1o, 16, ph1, pw1, c2_w_i8, c2_w_scale, 32, 3, 1, c2_b, a_scale_conv1, 0,
              c2o, &h2, &w2, a_scale_conv2, a_zero_conv2);
    relu_i8(c2o, 32 * h2 * w2, (int8_t)a_zero_conv2);
    int8_t p2o[32 * 8 * 8];
    int ph2, pw2;
    maxpool_i8(c2o, 32, h2, w2, p2o, &ph2, &pw2);

    /* Flatten → FC1 (int8 çıkış) → ReLU */
    int8_t fc1o[64];
    fc_i8_act(p2o, 32 * 8 * 8, fc1_w_i8, fc1_w_scale, 64, fc1_b,
              a_scale_conv2, fc1o, (int8_t)a_zero_fc1, a_scale_fc1);
    relu_i8(fc1o, 64, (int8_t)a_zero_fc1);

    /* FC2 → float logit */
    float logits[4];
    fc_i8(fc1o, 64, fc2_w_i8, fc2_w_scale, 4, fc2_b, a_scale_fc1, logits);

    /* Softmax */
    double mx = -1e308;
    for (int c = 0; c < 4; c++) if (logits[c] > mx) mx = logits[c];
    double sum = 0.0;
    for (int c = 0; c < 4; c++) {
        logits[c] = (float)exp((double)logits[c] - mx);
        sum += logits[c];
    }
    for (int c = 0; c < 4; c++) logits[c] = (float)(logits[c] / sum);

    int best = 0;
    for (int c = 1; c < 4; c++) if (logits[c] > logits[best]) best = c;
    if (probs) {
        for (int c = 0; c < 4; c++) probs[c] = logits[c];
    }
    return best;
}
