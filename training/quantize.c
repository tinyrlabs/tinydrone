/**
 * quantize.c — float CNN → int8 quantization
 *
 * Per-output-channel ağırlık scale + per-tensor aktivasyon scale.
 * Kalibrasyon: train setinden örneklerle aktivasyon aralıkları ölçülür.
 *
 * Çıktı: output/tinydrone_model_int8.h
 *
 * Build:
 *   cc -std=c11 -O2 -I/path/to/tinycml/include -I. -Ioutput \
 *      quantize.c /path/to/tinycml/build/lib/libtinycml.a -lm -o quantize
 * Run:
 *   ./quantize dataset/processed
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dirent.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "matrix.h"
#include "conv2d.h"
#include "pool2d.h"
#include "tinydrone_model.h"  /* float ağırlıklar */

#define IMG_H 32
#define IMG_W 32
#define IMG_C 3
#define N_CLASSES 5
#define NCHW (IMG_C * IMG_H * IMG_W)

/* ---------- Katman yapıları ---------- */
#define C1_W 432  /* 16×27 */
#define C2_W 4608 /* 32×144 */
#define FC1_W 131072 /* 2048×64 */
#define FC2_W 256 /* 64×4 */

/* Ağırlık düzenleri (row-major, satır = output kanal) */
static const int C1_OUT = 16, C1_IN = 27;
static const int C2_OUT = 32, C2_IN = 144;
static const int FC1_OUT = 64, FC1_IN = 2048;
static const int FC2_OUT = 4, FC2_IN = 64;

/* ---------- int8 quantize yardımcıları ---------- */

typedef struct {
    double scale;
    int8_t zero;
} QScale;

/* Per-row (per-output-channel) min/max → scale/zero (simetrik) */
static QScale compute_row_scale(const double *row, int n) {
    double mn = 1e30, mx = -1e30;
    for (int i = 0; i < n; i++) {
        if (row[i] < mn) mn = row[i];
        if (row[i] > mx) mx = row[i];
    }
    QScale q;
    q.zero = 0;
    /* Simetrik: scale = max(|min|,|max|)/127 — int8'e tam oturur */
    double m = fmax(fabs(mn), fabs(mx));
    if (m < 1e-12) m = 1e-12;
    q.scale = m / 127.0;
    return q;
}

/* Per-tensor min/max → scale/zero */
static QScale compute_tensor_scale(const double *data, int n) {
    double mn = 1e30, mx = -1e30;
    for (int i = 0; i < n; i++) {
        if (data[i] < mn) mn = data[i];
        if (data[i] > mx) mx = data[i];
    }
    QScale q;
    q.zero = 0;
    double range = mx - mn;
    if (range < 1e-12) range = 1e-12;
    q.scale = range / 255.0;
    return q;
}

static int8_t q_round(double v, double scale) {
    double x = v / scale;
    if (x > 127.0) return 127;
    if (x < -128.0) return -128;
    return (int8_t)lround(x);
}

/* ---------- Aktivasyon aralığı ölçümü (kalibrasyon) ---------- */

typedef struct {
    double mn, mx;
} Range;

static void range_update(Range *r, const double *data, int n) {
    for (int i = 0; i < n; i++) {
        if (data[i] < r->mn) r->mn = data[i];
        if (data[i] > r->mx) r->mx = data[i];
    }
}

static void relu_inplace(Matrix *m) {
    for (size_t i = 0; i < m->rows * m->cols; i++)
        if (m->data[i] < 0.0) m->data[i] = 0.0;
}

/* Kalibrasyon: float forward ile her katman çıkış aralığını ölç */
static void calibrate(Conv2D *c1, Conv2D *c2, MaxPool2D *p1, MaxPool2D *p2,
                      Matrix *fc1_w, Matrix *fc1_b, Matrix *fc2_w, Matrix *fc2_b,
                      const char *data_root, int max_imgs,
                      Range *r_conv1_out, Range *r_conv2_out,
                      Range *r_fc1_out, Range *r_fc2_out) {
    const char *cls[] = {"tank", "armored_vehicle", "drone_uav", "background"};
    int n_imgs = 0;

    for (int c = 0; c < N_CLASSES && n_imgs < max_imgs; c++) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/train/%s", data_root, cls[c]);
        DIR *d = opendir(path);
        if (!d) continue;
        struct dirent *e;
        while ((e = readdir(d)) && n_imgs < max_imgs) {
            if (!strstr(e->d_name, ".png")) continue;
            char fp[2048];
            snprintf(fp, sizeof(fp), "%s/%s", path, e->d_name);
            int w, h, ch;
            unsigned char *px = stbi_load(fp, &w, &h, &ch, 3);
            if (!px) continue;

            Matrix *X = matrix_alloc(1, NCHW);
            for (int y = 0; y < IMG_H; y++)
                for (int x = 0; x < IMG_W; x++)
                    for (int cc = 0; cc < IMG_C; cc++)
                        X->data[cc * IMG_H * IMG_W + y * IMG_W + x] =
                            (double)px[(y * w + x) * 3 + cc] / 127.5 - 1.0;
            stbi_image_free(px);

            Matrix *a1 = conv2d_forward(c1, X, 1, IMG_H, IMG_W);
            range_update(r_conv1_out, a1->data, a1->rows * a1->cols);
            Matrix *a1r = matrix_copy(a1); relu_inplace(a1r); matrix_free(a1);
            Matrix *a1p = maxpool2d_forward(p1, a1r, 1, 16, IMG_H, IMG_W); matrix_free(a1r);

            Matrix *a2 = conv2d_forward(c2, a1p, 1, 16, 16);
            range_update(r_conv2_out, a2->data, a2->rows * a2->cols);
            Matrix *a2r = matrix_copy(a2); relu_inplace(a2r); matrix_free(a2);
            Matrix *a2p = maxpool2d_forward(p2, a2r, 1, 32, 16, 16); matrix_free(a2r);

            Matrix *fc1 = matrix_matmul(a2p, fc1_w); matrix_free(a2p);
            for (int j = 0; j < 64; j++) fc1->data[j] += fc1_b->data[j];
            relu_inplace(fc1);
            range_update(r_fc1_out, fc1->data, fc1->rows * fc1->cols);

            Matrix *out = matrix_matmul(fc1, fc2_w); matrix_free(fc1);
            for (int j = 0; j < N_CLASSES; j++) out->data[j] += fc2_b->data[j];
            range_update(r_fc2_out, out->data, out->rows * out->cols);
            matrix_free(out);

            matrix_free(X);
            n_imgs++;
        }
        closedir(d);
    }
    printf("Kalibrasyon: %d görsel\n", n_imgs);
}

/* ---------- int8 header yaz ---------- */

static void write_header(const char *path,
                         const int8_t *c1q, QScale *c1_s, int8_t *c1bq, QScale c1b_s,
                         const int8_t *c2q, QScale *c2_s, int8_t *c2bq, QScale c2b_s,
                         const int8_t *fc1q, QScale *fc1_s, int8_t *fc1bq, QScale fc1b_s,
                         const int8_t *fc2q, QScale *fc2_s, int8_t *fc2bq, QScale fc2b_s,
                         Range *r_conv1, Range *r_conv2, Range *r_fc1, Range *r_fc2) {
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path); return; }

    fprintf(f, "/**\n * tinydrone_model_int8.h — int8 quantize edilmiş model\n");
    fprintf(f, " * Auto-generated by quantize.c. Do not edit.\n */\n");
    fprintf(f, "#ifndef TINYDRONE_MODEL_INT8_H\n#define TINYDRONE_MODEL_INT8_H\n\n");
    fprintf(f, "#include <stdint.h>\n\n");
    fprintf(f, "#define TD_I8_N_CLASSES %d\n\n", N_CLASSES);

    /* Aktivasyon aralıkları (kalibrasyon) */
    fprintf(f, "/* Aktivasyon aralıkları (kalibrasyon seti) */\n");
    fprintf(f, "static const float act_conv1_min = %.6ff, act_conv1_max = %.6ff;\n", r_conv1->mn, r_conv1->mx);
    fprintf(f, "static const float act_conv2_min = %.6ff, act_conv2_max = %.6ff;\n", r_conv2->mn, r_conv2->mx);
    fprintf(f, "static const float act_fc1_min = %.6ff, act_fc1_max = %.6ff;\n", r_fc1->mn, r_fc1->mx);
    fprintf(f, "static const float act_fc2_min = %.6ff, act_fc2_max = %.6ff;\n\n", r_fc2->mn, r_fc2->mx);

    /* Conv1 */
    fprintf(f, "static const int8_t c1_w_i8[%d] = {", C1_W);
    for (int i = 0; i < C1_W; i++) fprintf(f, "%s%d", i ? "," : "", c1q[i]);
    fprintf(f, "};\nstatic const float c1_w_scale[%d] = {", C1_OUT);
    for (int i = 0; i < C1_OUT; i++) fprintf(f, "%s%.8g", i ? "," : "", c1_s[i].scale);
    fprintf(f, "};\nstatic const int8_t c1_b_i8[%d] = {", C1_OUT);
    for (int i = 0; i < C1_OUT; i++) fprintf(f, "%s%d", i ? "," : "", c1bq[i]);
    fprintf(f, "};\nstatic const float c1_b_scale = %.8g;\n\n", c1b_s.scale);

    /* Conv2 */
    fprintf(f, "static const int8_t c2_w_i8[%d] = {", C2_W);
    for (int i = 0; i < C2_W; i++) fprintf(f, "%s%d", i ? "," : "", c2q[i]);
    fprintf(f, "};\nstatic const float c2_w_scale[%d] = {", C2_OUT);
    for (int i = 0; i < C2_OUT; i++) fprintf(f, "%s%.8g", i ? "," : "", c2_s[i].scale);
    fprintf(f, "};\nstatic const int8_t c2_b_i8[%d] = {", C2_OUT);
    for (int i = 0; i < C2_OUT; i++) fprintf(f, "%s%d", i ? "," : "", c2bq[i]);
    fprintf(f, "};\nstatic const float c2_b_scale = %.8g;\n\n", c2b_s.scale);

    /* FC1 */
    fprintf(f, "static const int8_t fc1_w_i8[%d] = {", FC1_W);
    for (int i = 0; i < FC1_W; i++) fprintf(f, "%s%d", i ? "," : "", fc1q[i]);
    fprintf(f, "};\nstatic const float fc1_w_scale[%d] = {", FC1_OUT);
    for (int i = 0; i < FC1_OUT; i++) fprintf(f, "%s%.8g", i ? "," : "", fc1_s[i].scale);
    fprintf(f, "};\nstatic const int8_t fc1_b_i8[%d] = {", FC1_OUT);
    for (int i = 0; i < FC1_OUT; i++) fprintf(f, "%s%d", i ? "," : "", fc1bq[i]);
    fprintf(f, "};\nstatic const float fc1_b_scale = %.8g;\n\n", fc1b_s.scale);

    /* FC2 */
    fprintf(f, "static const int8_t fc2_w_i8[%d] = {", FC2_W);
    for (int i = 0; i < FC2_W; i++) fprintf(f, "%s%d", i ? "," : "", fc2q[i]);
    fprintf(f, "};\nstatic const float fc2_w_scale[%d] = {", FC2_OUT);
    for (int i = 0; i < FC2_OUT; i++) fprintf(f, "%s%.8g", i ? "," : "", fc2_s[i].scale);
    fprintf(f, "};\nstatic const int8_t fc2_b_i8[%d] = {", FC2_OUT);
    for (int i = 0; i < FC2_OUT; i++) fprintf(f, "%s%d", i ? "," : "", fc2bq[i]);
    fprintf(f, "};\nstatic const float fc2_b_scale = %.8g;\n\n", fc2b_s.scale);

    fprintf(f, "#endif /* TINYDRONE_MODEL_INT8_H */\n");
    fclose(f);
    printf("int8 model yazıldı: %s\n", path);
}

/* ---------- Ana ---------- */

int main(int argc, char **argv) {
    const char *data_root = argc > 1 ? argv[1] : "dataset/processed";
    const char *out_path = argc > 2 ? argv[2] : "output/tinydrone_model_int8.h";
    int calib_imgs = argc > 3 ? atoi(argv[3]) : 200;

    srand((unsigned)time(NULL));
    printf("tinydrone — int8 Quantization\n");
    printf("==============================\n");

    /* Float model kur */
    Conv2D *c1 = conv2d_create_with_weights(3, 16, 3, 3, 1, 1, 1, 1, c1_w, c1_b);
    Conv2D *c2 = conv2d_create_with_weights(16, 32, 3, 3, 1, 1, 1, 1, c2_w, c2_b);
    MaxPool2D *p1 = maxpool2d_create(2, 2, 2, 2);
    MaxPool2D *p2 = maxpool2d_create(2, 2, 2, 2);
    Matrix *m_fc1_w = matrix_alloc(FC1_IN, FC1_OUT);
    Matrix *m_fc1_b = matrix_alloc(1, FC1_OUT);
    Matrix *m_fc2_w = matrix_alloc(FC2_IN, FC2_OUT);
    Matrix *m_fc2_b = matrix_alloc(1, FC2_OUT);
    memcpy(m_fc1_w->data, fc1_w, FC1_W * sizeof(double));
    memcpy(m_fc1_b->data, fc1_b, FC1_OUT * sizeof(double));
    memcpy(m_fc2_w->data, fc2_w, FC2_W * sizeof(double));
    memcpy(m_fc2_b->data, fc2_b, FC2_OUT * sizeof(double));

    /* Kalibrasyon */
    Range r1 = {1e30, -1e30}, r2 = {1e30, -1e30}, r3 = {1e30, -1e30}, r4 = {1e30, -1e30};
    calibrate(c1, c2, p1, p2, m_fc1_w, m_fc1_b, m_fc2_w, m_fc2_b, data_root, calib_imgs,
              &r1, &r2, &r3, &r4);
    printf("Aktivasyon aralıkları:\n");
    printf("  conv1: [%.4f, %.4f]\n", r1.mn, r1.mx);
    printf("  conv2: [%.4f, %.4f]\n", r2.mn, r2.mx);
    printf("  fc1:   [%.4f, %.4f]\n", r3.mn, r3.mx);
    printf("  fc2:   [%.4f, %.4f]\n\n", r4.mn, r4.mx);

    /* Ağırlık quantize — per-output-channel */
    QScale c1_s[C1_OUT], c2_s[C2_OUT], fc1_s[FC1_OUT], fc2_s[FC2_OUT];
    int8_t c1q[C1_W], c2q[C2_W], fc1q[FC1_W], fc2q[FC2_W];
    int8_t c1bq[C1_OUT], c2bq[C2_OUT], fc1bq[FC1_OUT], fc2bq[FC2_OUT];

    for (int o = 0; o < C1_OUT; o++) {
        c1_s[o] = compute_row_scale(&c1_w[o * C1_IN], C1_IN);
        for (int i = 0; i < C1_IN; i++) c1q[o * C1_IN + i] = q_round(c1_w[o * C1_IN + i], c1_s[o].scale);
        c1bq[o] = q_round(c1_b[o], c1_s[o].scale);  /* bias aynı scale */
    }
    for (int o = 0; o < C2_OUT; o++) {
        c2_s[o] = compute_row_scale(&c2_w[o * C2_IN], C2_IN);
        for (int i = 0; i < C2_IN; i++) c2q[o * C2_IN + i] = q_round(c2_w[o * C2_IN + i], c2_s[o].scale);
        c2bq[o] = q_round(c2_b[o], c2_s[o].scale);
    }
    /* FC1: input-major düzen (tinycml: 2048×64, w[i*64+o]) */
    for (int o = 0; o < FC1_OUT; o++) {
        /* per-channel scale: output nöronu o için stride'lı topla */
        double mn = 1e30, mx = -1e30;
        for (int i = 0; i < FC1_IN; i++) {
            double v = fc1_w[i * FC1_OUT + o];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        double m = fmax(fabs(mn), fabs(mx));
        if (m < 1e-12) m = 1e-12;
        fc1_s[o].scale = m / 127.0;
        fc1_s[o].zero = 0;
        for (int i = 0; i < FC1_IN; i++)
            fc1q[i * FC1_OUT + o] = q_round(fc1_w[i * FC1_OUT + o], fc1_s[o].scale);
        fc1bq[o] = q_round(fc1_b[o], fc1_s[o].scale);
    }
    /* FC2: input-major düzen (64×4, w[i*4+o]) */
    for (int o = 0; o < FC2_OUT; o++) {
        double mn = 1e30, mx = -1e30;
        for (int i = 0; i < FC2_IN; i++) {
            double v = fc2_w[i * FC2_OUT + o];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        double m = fmax(fabs(mn), fabs(mx));
        if (m < 1e-12) m = 1e-12;
        fc2_s[o].scale = m / 127.0;
        fc2_s[o].zero = 0;
        for (int i = 0; i < FC2_IN; i++)
            fc2q[i * FC2_OUT + o] = q_round(fc2_w[i * FC2_OUT + o], fc2_s[o].scale);
        fc2bq[o] = q_round(fc2_b[o], fc2_s[o].scale);
    }

    QScale c1b_s = compute_tensor_scale(c1_b, C1_OUT);
    QScale c2b_s = compute_tensor_scale(c2_b, C2_OUT);
    QScale fc1b_s = compute_tensor_scale(fc1_b, FC1_OUT);
    QScale fc2b_s = compute_tensor_scale(fc2_b, FC2_OUT);

    write_header(out_path, c1q, c1_s, c1bq, c1b_s,
                 c2q, c2_s, c2bq, c2b_s,
                 fc1q, fc1_s, fc1bq, fc1b_s,
                 fc2q, fc2_s, fc2bq, fc2b_s,
                 &r1, &r2, &r3, &r4);

    /* Temizlik */
    conv2d_free(c1); conv2d_free(c2);
    maxpool2d_free(p1); maxpool2d_free(p2);
    matrix_free(m_fc1_w); matrix_free(m_fc1_b);
    matrix_free(m_fc2_w); matrix_free(m_fc2_b);

    return 0;
}
