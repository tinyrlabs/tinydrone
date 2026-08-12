/**
 * train_tinydrone.c — tinycml CNN training for military target detection
 *
 * Build:
 *   cc -std=c11 -O2 -I/path/to/tinycml/include -I. \
 *      train_tinydrone.c /path/to/tinycml/build/lib/libtinycml.a -lm -o train_tinydrone
 *
 * Run:
 *   ./train_tinydrone [dataset_dir] [epochs] [batch_size] [learning_rate]
 *
 * Architecture (verified):
 *   Conv2D(3→16, 3x3, pad=1) → ReLU → MaxPool(2x2)   # 32→16, 16ch
 *   Conv2D(16→32, 3x3, pad=1) → ReLU → MaxPool(2x2)  # 16→8, 32ch
 *   Flatten → 32*8*8 = 2048
 *   Dense(2048→64) → ReLU → Dense(64→4) → Softmax
 *
 * Uses tinycml's own Conv2D/MaxPool2D forward+backward — zero Python.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dirent.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "conv2d.h"
#include "pool2d.h"
#include "matrix.h"

#define IMG_H 32
#define IMG_W 32
#define IMG_C 3
#define N_CLASSES 4
#define NCHW (IMG_C * IMG_H * IMG_W)  /* 3072 */

static const char *CLASS_NAMES[N_CLASSES] = {
    "tank", "armored_vehicle", "drone_uav", "background"
};

/* ============================================
 * Sample
 * ============================================ */

typedef struct {
    Matrix *img;   /* (1 × 3072), normalized to [-1, 1] */
    int label;
} Sample;

static void relu_inplace(Matrix *m) {
    for (size_t i = 0; i < m->rows * m->cols; i++)
        if (m->data[i] < 0.0) m->data[i] = 0.0;
}

static void softmax_rows(Matrix *m, int n) {
    for (int r = 0; r < n; r++) {
        double mx = -1e308;
        for (int c = 0; c < N_CLASSES; c++)
            if (m->data[r * N_CLASSES + c] > mx) mx = m->data[r * N_CLASSES + c];
        double sum = 0.0;
        for (int c = 0; c < N_CLASSES; c++) {
            m->data[r * N_CLASSES + c] = exp(m->data[r * N_CLASSES + c] - mx);
            sum += m->data[r * N_CLASSES + c];
        }
        for (int c = 0; c < N_CLASSES; c++)
            m->data[r * N_CLASSES + c] /= sum;
    }
}

/* ============================================
 * Dataset loading
 * ============================================ */

static Sample* load_dataset(const char *root, int max_per_class, int *out_n) {
    Sample *samples = NULL;
    int n = 0;

    for (int c = 0; c < N_CLASSES; c++) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/train/%s", root, CLASS_NAMES[c]);
        DIR *d = opendir(path);
        if (!d) {
            fprintf(stderr, "  warn: cannot open %s\n", path);
            continue;
        }
        struct dirent *e;
        int cnt = 0;
        while ((e = readdir(d)) && cnt < max_per_class) {
            if (!strstr(e->d_name, ".png")) continue;
            char fp[2048];
            snprintf(fp, sizeof(fp), "%s/%s", path, e->d_name);
            int w, h, ch;
            unsigned char *px = stbi_load(fp, &w, &h, &ch, 3);
            if (!px) continue;

            Matrix *img = matrix_alloc(1, NCHW);
            if (!img) { stbi_image_free(px); continue; }
            for (int y = 0; y < IMG_H; y++)
                for (int x = 0; x < IMG_W; x++)
                    for (int cc = 0; cc < IMG_C; cc++)
                        img->data[cc * IMG_H * IMG_W + y * IMG_W + x] =
                            (double)px[(y * w + x) * 3 + cc] / 127.5 - 1.0;
            stbi_image_free(px);

            samples = realloc(samples, (size_t)(n + 1) * sizeof(Sample));
            samples[n].img = img;
            samples[n].label = c;
            n++;
            cnt++;
        }
        closedir(d);
    }

    *out_n = n;
    return samples;
}

/* ============================================
 * Model (explicit layers)
 * ============================================ */

typedef struct {
    Conv2D *c1, *c2;
    MaxPool2D *p1, *p2;
    Matrix *fc1_w, *fc1_b;   /* (2048 × 64), (1 × 64) */
    Matrix *fc2_w, *fc2_b;   /* (64 × 4), (1 × 4) */
} TinyCNN;

static TinyCNN* model_create(void) {
    TinyCNN *m = calloc(1, sizeof(TinyCNN));
    m->c1 = conv2d_create(3, 16, 3, 3, 1, 1, 1, 1);
    m->p1 = maxpool2d_create(2, 2, 2, 2);
    m->c2 = conv2d_create(16, 32, 3, 3, 1, 1, 1, 1);
    m->p2 = maxpool2d_create(2, 2, 2, 2);

    int flat = 32 * 8 * 8;  /* 2048 */
    m->fc1_w = matrix_alloc(flat, 64);
    m->fc1_b = matrix_alloc(1, 64);
    m->fc2_w = matrix_alloc(64, N_CLASSES);
    m->fc2_b = matrix_alloc(1, N_CLASSES);

    /* Xavier-ish init */
    double lim1 = sqrt(6.0 / (flat + 64));
    double lim2 = sqrt(6.0 / (64 + N_CLASSES));
    for (size_t i = 0; i < m->fc1_w->rows * m->fc1_w->cols; i++)
        m->fc1_w->data[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * lim1;
    for (size_t i = 0; i < m->fc2_w->rows * m->fc2_w->cols; i++)
        m->fc2_w->data[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * lim2;

    return m;
}

static void model_free(TinyCNN *m) {
    if (!m) return;
    conv2d_free(m->c1); maxpool2d_free(m->p1);
    conv2d_free(m->c2); maxpool2d_free(m->p2);
    matrix_free(m->fc1_w); matrix_free(m->fc1_b);
    matrix_free(m->fc2_w); matrix_free(m->fc2_b);
    free(m);
}

/* ============================================
 * Forward pass (returns final output; intermediates freed here)
 * ============================================ */

static Matrix* forward(TinyCNN *m, const Matrix *X, int bn) {
    Matrix *a1 = conv2d_forward(m->c1, X, bn, IMG_H, IMG_W);
    Matrix *a1r = matrix_copy(a1); relu_inplace(a1r); matrix_free(a1);
    Matrix *a1p = maxpool2d_forward(m->p1, a1r, bn, 16, IMG_H, IMG_W); matrix_free(a1r);

    Matrix *a2 = conv2d_forward(m->c2, a1p, bn, 16, 16); matrix_free(a1p);
    Matrix *a2r = matrix_copy(a2); relu_inplace(a2r); matrix_free(a2);
    Matrix *a2p = maxpool2d_forward(m->p2, a2r, bn, 32, 16, 16); matrix_free(a2r);

    Matrix *fc1 = matrix_matmul(a2p, m->fc1_w); matrix_free(a2p);
    for (int r = 0; r < bn; r++)
        for (int c = 0; c < 64; c++)
            fc1->data[r * 64 + c] += m->fc1_b->data[c];
    relu_inplace(fc1);

    Matrix *out = matrix_matmul(fc1, m->fc2_w); matrix_free(fc1);
    for (int r = 0; r < bn; r++)
        for (int c = 0; c < N_CLASSES; c++)
            out->data[r * N_CLASSES + c] += m->fc2_b->data[c];
    softmax_rows(out, bn);

    return out;
}

/* ============================================
 * Backward pass (SGD update in-place)
 * ============================================ */

static void backward(TinyCNN *m, const Matrix *X, const Matrix *out,
                     const int *labels, int bn, double lr) {
    /* Forward pass — keep ALL intermediates alive until backward is done.
     * Freed at the end: a1r, a1p, a2r, a2p, fc1. */
    Matrix *a1  = conv2d_forward(m->c1, X, bn, IMG_H, IMG_W);
    Matrix *a1r = matrix_copy(a1); relu_inplace(a1r); matrix_free(a1);
    Matrix *a1p = maxpool2d_forward(m->p1, a1r, bn, 16, IMG_H, IMG_W);
    Matrix *a2  = conv2d_forward(m->c2, a1p, bn, 16, 16);
    Matrix *a2r = matrix_copy(a2); relu_inplace(a2r); matrix_free(a2);
    Matrix *a2p = maxpool2d_forward(m->p2, a2r, bn, 32, 16, 16);
    Matrix *fc1 = matrix_matmul(a2p, m->fc1_w);
    for (int r = 0; r < bn; r++)
        for (int c = 0; c < 64; c++)
            fc1->data[r * 64 + c] += m->fc1_b->data[c];
    relu_inplace(fc1);

    /* dL/dout = softmax_out - onehot */
    Matrix *dout = matrix_copy(out);
    for (int i = 0; i < bn; i++)
        dout->data[i * N_CLASSES + labels[i]] -= 1.0;

    /* FC2 gradients */
    Matrix *fc1T = matrix_transpose(fc1);
    Matrix *dw2  = matrix_matmul(fc1T, dout); matrix_free(fc1T);
    Matrix *w2T  = matrix_transpose(m->fc2_w);
    Matrix *dfc1 = matrix_matmul(dout, w2T); matrix_free(w2T);
    for (size_t j = 0; j < m->fc2_w->rows * m->fc2_w->cols; j++)
        m->fc2_w->data[j] -= lr * dw2->data[j];
    for (size_t c = 0; c < m->fc2_b->cols; c++) {
        double s = 0;
        for (size_t r = 0; r < dout->rows; r++) s += dout->data[r * m->fc2_b->cols + c];
        m->fc2_b->data[c] -= lr * s;
    }
    matrix_free(dw2);

    /* ReLU(FC1) backward — mask from fc1 pre-activation */
    for (size_t j = 0; j < fc1->rows * fc1->cols; j++)
        if (fc1->data[j] <= 0) dfc1->data[j] = 0;

    /* FC1 gradients */
    Matrix *a2pT = matrix_transpose(a2p);
    Matrix *dw1  = matrix_matmul(a2pT, dfc1); matrix_free(a2pT);
    Matrix *w1T  = matrix_transpose(m->fc1_w);
    Matrix *da2p = matrix_matmul(dfc1, w1T); matrix_free(w1T);
    for (size_t j = 0; j < m->fc1_w->rows * m->fc1_w->cols; j++)
        m->fc1_w->data[j] -= lr * dw1->data[j];
    for (size_t c = 0; c < m->fc1_b->cols; c++) {
        double s = 0;
        for (size_t r = 0; r < dfc1->rows; r++) s += dfc1->data[r * m->fc1_b->cols + c];
        m->fc1_b->data[c] -= lr * s;
    }
    matrix_free(dw1);

    /* Pool2 / ReLU2 / Conv2 */
    Matrix *da2r = maxpool2d_backward(m->p2, da2p);
    for (size_t j = 0; j < da2r->rows * da2r->cols; j++)
        if (a2r->data[j] <= 0) da2r->data[j] = 0;
    Matrix *da1p = conv2d_backward(m->c2, da2r);
    conv2d_update_weights(m->c2, lr);
    matrix_free(da2r); matrix_free(da2p); matrix_free(dfc1);

    /* Pool1 / ReLU1 / Conv1 */
    Matrix *da1r = maxpool2d_backward(m->p1, da1p);
    for (size_t j = 0; j < da1r->rows * da1r->cols; j++)
        if (a1r->data[j] <= 0) da1r->data[j] = 0;
    Matrix *dX = conv2d_backward(m->c1, da1r);
    conv2d_update_weights(m->c1, lr);
    matrix_free(da1r); matrix_free(da1p); matrix_free(dX);

    /* Free forward intermediates */
    matrix_free(dout);
    matrix_free(a1r); matrix_free(a1p);
    matrix_free(a2r); matrix_free(a2p);
    matrix_free(fc1);
}

/* ============================================
 * Weight export (C header for firmware)
 * ============================================ */

static void export_weights(TinyCNN *m, const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return; }

    fprintf(f, "/**\n * tinydrone_model.h — auto-generated by train_tinydrone.c\n */\n");
    fprintf(f, "#ifndef TINYDRONE_MODEL_H\n#define TINYDRONE_MODEL_H\n\n");
    fprintf(f, "#define TD_N_CLASSES %d\n\n", N_CLASSES);

    /* Conv layers */
    fprintf(f, "static const double c1_w[%zu] = {", m->c1->weight->rows * m->c1->weight->cols);
    for (size_t i = 0; i < m->c1->weight->rows * m->c1->weight->cols; i++)
        fprintf(f, "%s%.8g", i ? "," : "", m->c1->weight->data[i]);
    fprintf(f, "};\nstatic const double c1_b[%zu] = {", m->c1->bias->cols);
    for (size_t i = 0; i < m->c1->bias->cols; i++)
        fprintf(f, "%s%.8g", i ? "," : "", m->c1->bias->data[i]);
    fprintf(f, "};\n\n");

    fprintf(f, "static const double c2_w[%zu] = {", m->c2->weight->rows * m->c2->weight->cols);
    for (size_t i = 0; i < m->c2->weight->rows * m->c2->weight->cols; i++)
        fprintf(f, "%s%.8g", i ? "," : "", m->c2->weight->data[i]);
    fprintf(f, "};\nstatic const double c2_b[%zu] = {", m->c2->bias->cols);
    for (size_t i = 0; i < m->c2->bias->cols; i++)
        fprintf(f, "%s%.8g", i ? "," : "", m->c2->bias->data[i]);
    fprintf(f, "};\n\n");

    /* Dense layers */
    fprintf(f, "static const double fc1_w[%zu] = {", m->fc1_w->rows * m->fc1_w->cols);
    for (size_t i = 0; i < m->fc1_w->rows * m->fc1_w->cols; i++)
        fprintf(f, "%s%.8g", i ? "," : "", m->fc1_w->data[i]);
    fprintf(f, "};\nstatic const double fc1_b[%zu] = {", m->fc1_b->cols);
    for (size_t i = 0; i < m->fc1_b->cols; i++)
        fprintf(f, "%s%.8g", i ? "," : "", m->fc1_b->data[i]);
    fprintf(f, "};\n\n");

    fprintf(f, "static const double fc2_w[%zu] = {", m->fc2_w->rows * m->fc2_w->cols);
    for (size_t i = 0; i < m->fc2_w->rows * m->fc2_w->cols; i++)
        fprintf(f, "%s%.8g", i ? "," : "", m->fc2_w->data[i]);
    fprintf(f, "};\nstatic const double fc2_b[%zu] = {", m->fc2_b->cols);
    for (size_t i = 0; i < m->fc2_b->cols; i++)
        fprintf(f, "%s%.8g", i ? "," : "", m->fc2_b->data[i]);
    fprintf(f, "};\n\n#endif\n");

    fclose(f);
    printf("Weights exported: %s\n", path);
}

/* ============================================
 * Main
 * ============================================ */

int main(int argc, char **argv) {
    const char *data_root = argc > 1 ? argv[1] : "dataset/processed";
    int epochs = argc > 2 ? atoi(argv[2]) : 20;
    int batch = argc > 3 ? atoi(argv[3]) : 8;
    double lr = argc > 4 ? atof(argv[4]) : 0.001;
    int max_per_class = argc > 5 ? atoi(argv[5]) : 0;  /* 0 = all */

    srand((unsigned)time(NULL));

    printf("tinydrone — tinycml CNN training\n");
    printf("================================\n");
    printf("Data: %s | Epochs: %d | Batch: %d | LR: %.4f\n\n",
           data_root, epochs, batch, lr);

    /* Load dataset */
    printf("Loading dataset...\n"); fflush(stdout);
    int n = 0;
    Sample *samples = load_dataset(data_root, max_per_class ? max_per_class : 1 << 30, &n);
    if (n == 0) { fprintf(stderr, "No data found!\n"); return 1; }
    printf("  Loaded %d samples\n\n", n); fflush(stdout);

    /* Model */
    TinyCNN *model = model_create();
    printf("Model: Conv(16)-ReLU-Pool | Conv(32)-ReLU-Pool | FC(64)-FC(4)\n");
    printf("Params: ~%zu\n\n",
           model->c1->weight->rows * model->c1->weight->cols +
           model->c2->weight->rows * model->c2->weight->cols +
           model->fc1_w->rows * model->fc1_w->cols +
           model->fc2_w->rows * model->fc2_w->cols);
    fflush(stdout);

    /* Training loop */
    printf("Training %d epochs...\n", epochs); fflush(stdout);
    for (int ep = 0; ep < epochs; ep++) {
        /* Shuffle */
        for (int i = n - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            Sample t = samples[i]; samples[i] = samples[j]; samples[j] = t;
        }

        double loss_sum = 0.0;
        int batches = 0;

        for (int start = 0; start < n; start += batch) {
            int bn = (start + batch <= n) ? batch : (n - start);
            if (bn < 2) continue;

            /* Build batch */
            Matrix *X = matrix_alloc(bn, NCHW);
            int *labels = malloc((size_t)bn * sizeof(int));
            for (int b = 0; b < bn; b++) {
                memcpy(&X->data[b * NCHW], samples[start + b].img->data,
                       NCHW * sizeof(double));
                labels[b] = samples[start + b].label;
            }

            /* Forward + loss */
            Matrix *out = forward(model, X, bn);
            double loss = 0.0;
            for (int i = 0; i < bn; i++)
                loss -= log(out->data[i * N_CLASSES + labels[i]] + 1e-8);
            loss /= bn;
            loss_sum += loss;

            /* Backward */
            backward(model, X, out, labels, bn, lr);

            /* Cleanup */
            matrix_free(X); matrix_free(out);
            free(labels);
            batches++;
        }

        printf("Epoch %3d/%d | Loss: %.4f | Batches: %d\n",
               ep + 1, epochs, loss_sum / batches, batches);
        fflush(stdout);

        /* Snapshot weights after each epoch */
        char snap[512];
        snprintf(snap, sizeof(snap), "output/weights_epoch%02d.h", ep + 1);
        export_weights(model, snap);
    }

    /* Final model */
    export_weights(model, "output/tinydrone_model.h");

    /* Cleanup */
    for (int i = 0; i < n; i++) matrix_free(samples[i].img);
    free(samples);
    model_free(model);

    printf("\nTraining complete. Weights in output/tinydrone_model.h\n");
    return 0;
}
