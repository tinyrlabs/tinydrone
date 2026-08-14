/**
 * inference.c — tinycml CNN inference (ESP32)
 *
 * ESP32 SRAM sınırlı (520KB). Model ~136K param × 8B = 1.1MB ağırlık
 * flash'ta (const). Çalışma zamanı buffer'ları küçük tutulur.
 *
 * NOT: Bu sürüm double (float64) kullanır — ESP32'de ~300-800ms/inference.
 *       Gerçek drone için int8 quantization önerilir (Faz 4.5).
 */

#include "float_inference.h"
#include <math.h>
#include <string.h>

#include "matrix.h"
#include "conv2d.h"
#include "pool2d.h"
#include "tinydrone_model.h"  /* tinydrone_model.h — ağırlıklar */

#define IMG_H 32
#define IMG_W 32
#define IMG_C 3

/* Katmanlar (inference sırasında static — heap'te ağırlık yok) */
static Conv2D s_c1, s_c2;
static MaxPool2D s_p1, s_p2;

/* FC ağırlıkları doğrudan model.h array'lerine işaret eder */
static Matrix s_fc1_w, s_fc1_b, s_fc2_w, s_fc2_b;

static void relu_inplace(Matrix *m) {
    for (size_t i = 0; i < m->rows * m->cols; i++)
        if (m->data[i] < 0.0) m->data[i] = 0.0;
}

/* const ağırlık array'ini Matrix olarak sarmala (alloc yok) */
static Matrix wrap_matrix(const double *data, size_t rows, size_t cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.data = (double *)data;
    return m;
}

void inference_init(void) {
    /* Conv2D layer'larını ağırlıklarla kur (const flash verisi) */
    memset(&s_c1, 0, sizeof(s_c1));
    s_c1.in_channels = 3;
    s_c1.out_channels = 16;
    s_c1.kernel_h = 3; s_c1.kernel_w = 3;
    s_c1.stride_h = 1; s_c1.stride_w = 1;
    s_c1.pad_h = 1; s_c1.pad_w = 1;
    s_c1.weight = matrix_alloc(16, 432 / 16);
    memcpy(s_c1.weight->data, c1_w, 432 * sizeof(double));
    s_c1.bias = matrix_alloc(1, 16);
    memcpy(s_c1.bias->data, c1_b, 16 * sizeof(double));

    memset(&s_c2, 0, sizeof(s_c2));
    s_c2.in_channels = 16;
    s_c2.out_channels = 32;
    s_c2.kernel_h = 3; s_c2.kernel_w = 3;
    s_c2.stride_h = 1; s_c2.stride_w = 1;
    s_c2.pad_h = 1; s_c2.pad_w = 1;
    s_c2.weight = matrix_alloc(32, 4608 / 32);
    memcpy(s_c2.weight->data, c2_w, 4608 * sizeof(double));
    s_c2.bias = matrix_alloc(1, 32);
    memcpy(s_c2.bias->data, c2_b, 32 * sizeof(double));

    memset(&s_p1, 0, sizeof(s_p1));
    s_p1.pool_h = 2; s_p1.pool_w = 2;
    s_p1.stride_h = 2; s_p1.stride_w = 2;
    memset(&s_p2, 0, sizeof(s_p2));
    s_p2.pool_h = 2; s_p2.pool_w = 2;
    s_p2.stride_h = 2; s_p2.stride_w = 2;

    s_fc1_w = wrap_matrix(fc1_w, 2048, 64);
    s_fc1_b = wrap_matrix(fc1_b, 1, 64);
    s_fc2_w = wrap_matrix(fc2_w, 64, TD_N_CLASSES);
    s_fc2_b = wrap_matrix(fc2_b, 1, TD_N_CLASSES);
}

int inference_run(const double *input, double *probs) {
    /* 32x32x3 giriş → 3072 elemanlı Matrix (stack, veri kopyalanmaz) */
    Matrix X;
    X.rows = 1;
    X.cols = IMG_C * IMG_H * IMG_W;
    X.data = (double *)input;  /* read-only kullanım */

    /* Forward — her katman stack'te, ara buffer'lar heap'te (küçük) */
    Matrix *a1 = conv2d_forward(&s_c1, &X, 1, IMG_H, IMG_W);
    if (!a1) return -1;
    relu_inplace(a1);
    Matrix *a1p = maxpool2d_forward(&s_p1, a1, 1, 16, IMG_H, IMG_W);
    matrix_free(a1);

    Matrix *a2 = conv2d_forward(&s_c2, a1p, 1, 16, 16);
    matrix_free(a1p);
    if (!a2) return -1;
    relu_inplace(a2);
    Matrix *a2p = maxpool2d_forward(&s_p2, a2, 1, 32, 16, 16);
    matrix_free(a2);

    Matrix *fc1 = matrix_matmul(a2p, &s_fc1_w);
    matrix_free(a2p);
    if (!fc1) return -1;
    for (int c = 0; c < 64; c++) fc1->data[c] += s_fc1_b.data[c];
    relu_inplace(fc1);

    Matrix *out = matrix_matmul(fc1, &s_fc2_w);
    matrix_free(fc1);
    if (!out) return -1;
    for (int c = 0; c < TD_N_CLASSES; c++) out->data[c] += s_fc2_b.data[c];

    /* Softmax */
    double mx = -1e308;
    for (int c = 0; c < TD_N_CLASSES; c++)
        if (out->data[c] > mx) mx = out->data[c];
    double sum = 0.0;
    for (int c = 0; c < TD_N_CLASSES; c++) {
        out->data[c] = exp(out->data[c] - mx);
        sum += out->data[c];
    }
    for (int c = 0; c < TD_N_CLASSES; c++) out->data[c] /= sum;

    int best = 0;
    for (int c = 1; c < TD_N_CLASSES; c++)
        if (out->data[c] > out->data[best]) best = c;

    if (probs) memcpy(probs, out->data, TD_N_CLASSES * sizeof(double));
    matrix_free(out);
    return best;
}
