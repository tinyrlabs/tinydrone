/**
 * evaluate.c — test seti doğruluğu ölçer (tinycml CNN inference)
 *
 * Build:
 *   cc -std=c11 -O2 -I/path/to/tinycml/include -I. -Ioutput \
 *      evaluate.c /path/to/tinycml/build/lib/libtinycml.a -lm -o evaluate
 *
 * Run:
 *   ./evaluate dataset/processed
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
#include "tinydrone_model.h"  /* exported weights */

#define IMG_H 32
#define IMG_W 32
#define IMG_C 3
#define N_CLASSES 4
#define NCHW (IMG_C * IMG_H * IMG_W)

static const char *CLASS_NAMES[N_CLASSES] = {
    "tank", "armored_vehicle", "drone_uav", "background"
};

static void relu_inplace(Matrix *m) {
    for (size_t i = 0; i < m->rows * m->cols; i++)
        if (m->data[i] < 0.0) m->data[i] = 0.0;
}

static int softmax_argmax(Matrix *m, int row, int n) {
    int best = 0;
    double best_val = m->data[row * n];
    for (int c = 1; c < n; c++) {
        if (m->data[row * n + c] > best_val) {
            best_val = m->data[row * n + c];
            best = c;
        }
    }
    return best;
}

/* Build model from exported weights */
static void build_model(Conv2D **c1, Conv2D **c2, MaxPool2D **p1, MaxPool2D **p2,
                        Matrix **fc1_w_out, Matrix **fc1_b_out,
                        Matrix **fc2_w_out, Matrix **fc2_b_out) {
    *c1 = conv2d_create_with_weights(3, 16, 3, 3, 1, 1, 1, 1, c1_w, c1_b);
    *p1 = maxpool2d_create(2, 2, 2, 2);
    *c2 = conv2d_create_with_weights(16, 32, 3, 3, 1, 1, 1, 1, c2_w, c2_b);
    *p2 = maxpool2d_create(2, 2, 2, 2);

    *fc1_w_out = matrix_alloc(2048, 64);
    *fc1_b_out = matrix_alloc(1, 64);
    *fc2_w_out = matrix_alloc(64, N_CLASSES);
    *fc2_b_out = matrix_alloc(1, N_CLASSES);
    memcpy((*fc1_w_out)->data, fc1_w, 2048 * 64 * sizeof(double));
    memcpy((*fc1_b_out)->data, fc1_b, 64 * sizeof(double));
    memcpy((*fc2_w_out)->data, fc2_w, 64 * N_CLASSES * sizeof(double));
    memcpy((*fc2_b_out)->data, fc2_b, N_CLASSES * sizeof(double));
}

/* Single-image inference. Returns predicted class. */
static int predict_one(Conv2D *c1, Conv2D *c2, MaxPool2D *p1, MaxPool2D *p2,
                       Matrix *fc1_w, Matrix *fc1_b, Matrix *fc2_w, Matrix *fc2_b,
                       const unsigned char *px, int w, int h) {
    Matrix *X = matrix_alloc(1, NCHW);
    for (int y = 0; y < IMG_H; y++)
        for (int x = 0; x < IMG_W; x++)
            for (int cc = 0; cc < IMG_C; cc++)
                X->data[cc * IMG_H * IMG_W + y * IMG_W + x] =
                    (double)px[(y * w + x) * 3 + cc] / 127.5 - 1.0;

    /* Forward */
    Matrix *a1 = conv2d_forward(c1, X, 1, IMG_H, IMG_W);
    Matrix *a1r = matrix_copy(a1); relu_inplace(a1r); matrix_free(a1);
    Matrix *a1p = maxpool2d_forward(p1, a1r, 1, 16, IMG_H, IMG_W); matrix_free(a1r);
    Matrix *a2 = conv2d_forward(c2, a1p, 1, 16, 16); matrix_free(a1p);
    Matrix *a2r = matrix_copy(a2); relu_inplace(a2r); matrix_free(a2);
    Matrix *a2p = maxpool2d_forward(p2, a2r, 1, 32, 16, 16); matrix_free(a2r);
    Matrix *fc1 = matrix_matmul(a2p, fc1_w); matrix_free(a2p);
    for (int c = 0; c < 64; c++) fc1->data[c] += fc1_b->data[c];
    relu_inplace(fc1);
    Matrix *out = matrix_matmul(fc1, fc2_w); matrix_free(fc1);
    for (int c = 0; c < N_CLASSES; c++) out->data[c] += fc2_b->data[c];
    /* Softmax not needed for argmax */
    int pred = softmax_argmax(out, 0, N_CLASSES);

    matrix_free(X); matrix_free(out);
    return pred;
}

static int eval_dir(const char *path, int expected_class,
                    Conv2D *c1, Conv2D *c2, MaxPool2D *p1, MaxPool2D *p2,
                    Matrix *fc1_w, Matrix *fc1_b, Matrix *fc2_w, Matrix *fc2_b,
                    int *total_out, int *correct_out) {
    DIR *d = opendir(path);
    if (!d) return 0;
    struct dirent *e;
    int correct = 0, total = 0;
    while ((e = readdir(d))) {
        if (!strstr(e->d_name, ".png")) continue;
        char fp[2048];
        snprintf(fp, sizeof(fp), "%s/%s", path, e->d_name);
        int w, h, ch;
        unsigned char *px = stbi_load(fp, &w, &h, &ch, 3);
        if (!px) continue;
        int pred = predict_one(c1, c2, p1, p2, fc1_w, fc1_b, fc2_w, fc2_b, px, w, h);
        stbi_image_free(px);
        if (pred == expected_class) correct++;
        total++;
    }
    closedir(d);
    *total_out += total;
    *correct_out += correct;
    return total;
}

int main(int argc, char **argv) {
    const char *data_root = argc > 1 ? argv[1] : "dataset/processed";
    int max_imgs = argc > 2 ? atoi(argv[2]) : 0;  /* 0 = all */

    printf("tinydrone — Model Evaluation\n");
    printf("=============================\n");

    /* Build model from exported header weights */
    Conv2D *c1, *c2;
    MaxPool2D *p1, *p2;
    Matrix *fc1_w, *fc1_b, *fc2_w, *fc2_b;
    build_model(&c1, &c2, &p1, &p2, &fc1_w, &fc1_b, &fc2_w, &fc2_b);
    printf("Model loaded from tinydrone_model.h\n\n");

    /* Eval each class on test split */
    int total = 0, correct = 0;
    double start = (double)clock() / CLOCKS_PER_SEC;

    for (int c = 0; c < N_CLASSES; c++) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/test/%s", data_root, CLASS_NAMES[c]);
        int t = 0, k = 0;
        int cnt = eval_dir(path, c, c1, c2, p1, p2, fc1_w, fc1_b, fc2_w, fc2_b, &t, &k);
        double acc = t > 0 ? 100.0 * k / t : 0.0;
        printf("  %-16s %4d img | accuracy: %6.2f%% (%d/%d)\n",
               CLASS_NAMES[c], cnt, acc, k, t);
        total += t;
        correct += k;
    }

    double elapsed = (double)clock() / CLOCKS_PER_SEC - start;
    double overall = total > 0 ? 100.0 * correct / total : 0.0;

    printf("\n  OVERALL: %.2f%% (%d/%d)\n", overall, correct, total);
    printf("  Time: %.1f s (%.1f ms/img)\n", elapsed,
           total > 0 ? 1000.0 * elapsed / total : 0.0);

    /* Cleanup */
    conv2d_free(c1); conv2d_free(c2);
    maxpool2d_free(p1); maxpool2d_free(p2);
    matrix_free(fc1_w); matrix_free(fc1_b);
    matrix_free(fc2_w); matrix_free(fc2_b);

    return 0;
}
