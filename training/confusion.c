/**
 * confusion.c — confusion matrix + per-class analiz
 *
 * Model davranışını anlamak: hangi sınıflar birbirine karışıyor?
 *
 * Build:
 *   cc -std=c11 -O2 -I/path/to/tinycml/include -I. -Ioutput \
 *      confusion.c /path/to/tinycml/build/lib/libtinycml.a -lm -o confusion
 *
 * Run:
 *   ./confusion dataset/processed
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
#include "tinydrone_model.h"

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

/* Model yapısı */
static Conv2D *c1, *c2;
static MaxPool2D *p1, *p2;
static Matrix *m_fc1_w, *m_fc1_b, *m_fc2_w, *m_fc2_b;

static void build_model(void) {
    c1 = conv2d_create_with_weights(3, 16, 3, 3, 1, 1, 1, 1, c1_w, c1_b);
    p1 = maxpool2d_create(2, 2, 2, 2);
    c2 = conv2d_create_with_weights(16, 32, 3, 3, 1, 1, 1, 1, c2_w, c2_b);
    p2 = maxpool2d_create(2, 2, 2, 2);
    m_fc1_w = matrix_alloc(2048, 64);
    m_fc1_b = matrix_alloc(1, 64);
    m_fc2_w = matrix_alloc(64, N_CLASSES);
    m_fc2_b = matrix_alloc(1, N_CLASSES);
    memcpy(m_fc1_w->data, fc1_w, 2048 * 64 * sizeof(double));
    memcpy(m_fc1_b->data, fc1_b, 64 * sizeof(double));
    memcpy(m_fc2_w->data, fc2_w, 64 * N_CLASSES * sizeof(double));
    memcpy(m_fc2_b->data, fc2_b, N_CLASSES * sizeof(double));
}

static void free_model(void) {
    conv2d_free(c1); conv2d_free(c2);
    maxpool2d_free(p1); maxpool2d_free(p2);
    matrix_free(m_fc1_w); matrix_free(m_fc1_b);
    matrix_free(m_fc2_w); matrix_free(m_fc2_b);
}

/* Inference: probs + predicted class */
static void predict(const unsigned char *px, int w, int h,
                    double *probs_out, int *pred_out) {
    (void)h;  /* 32x32 sabit input */
    Matrix *X = matrix_alloc(1, NCHW);
    for (int y = 0; y < IMG_H; y++)
        for (int x = 0; x < IMG_W; x++)
            for (int cc = 0; cc < IMG_C; cc++)
                X->data[cc * IMG_H * IMG_W + y * IMG_W + x] =
                    (double)px[(y * w + x) * 3 + cc] / 127.5 - 1.0;

    Matrix *a1 = conv2d_forward(c1, X, 1, IMG_H, IMG_W);
    Matrix *a1r = matrix_copy(a1); relu_inplace(a1r); matrix_free(a1);
    Matrix *a1p = maxpool2d_forward(p1, a1r, 1, 16, IMG_H, IMG_W); matrix_free(a1r);
    Matrix *a2 = conv2d_forward(c2, a1p, 1, 16, 16); matrix_free(a1p);
    Matrix *a2r = matrix_copy(a2); relu_inplace(a2r); matrix_free(a2);
    Matrix *a2p = maxpool2d_forward(p2, a2r, 1, 32, 16, 16); matrix_free(a2r);
    Matrix *fc1 = matrix_matmul(a2p, m_fc1_w); matrix_free(a2p);
    for (int c = 0; c < 64; c++) fc1->data[c] += m_fc1_b->data[c];
    relu_inplace(fc1);
    Matrix *out = matrix_matmul(fc1, m_fc2_w); matrix_free(fc1);
    for (int c = 0; c < N_CLASSES; c++) out->data[c] += m_fc2_b->data[c];

    /* Softmax */
    double mx = -1e308;
    for (int c = 0; c < N_CLASSES; c++) if (out->data[c] > mx) mx = out->data[c];
    double sum = 0.0;
    for (int c = 0; c < N_CLASSES; c++) {
        out->data[c] = exp(out->data[c] - mx);
        sum += out->data[c];
    }
    for (int c = 0; c < N_CLASSES; c++) out->data[c] /= sum;

    int best = 0;
    for (int c = 1; c < N_CLASSES; c++) if (out->data[c] > out->data[best]) best = c;

    if (probs_out) memcpy(probs_out, out->data, N_CLASSES * sizeof(double));
    if (pred_out) *pred_out = best;

    matrix_free(X); matrix_free(out);
}

int main(int argc, char **argv) {
    const char *data_root = argc > 1 ? argv[1] : "dataset/processed";
    int max_per_class = argc > 2 ? atoi(argv[2]) : 300;  /* test hızı için */

    printf("tinydrone — Confusion Matrix\n");
    printf("============================\n");
    build_model();
    printf("Model loaded.\n\n");

    /* [gerçek][tahmin] matrisi */
    int cm[N_CLASSES][N_CLASSES] = {{0}};
    int total = 0, correct = 0;

    for (int actual = 0; actual < N_CLASSES; actual++) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/test/%s", data_root, CLASS_NAMES[actual]);
        DIR *d = opendir(path);
        if (!d) continue;
        struct dirent *e;
        int cnt = 0;
        while ((e = readdir(d)) && cnt < max_per_class) {
            if (!strstr(e->d_name, ".png")) continue;
            char fp[2048];
            snprintf(fp, sizeof(fp), "%s/%s", path, e->d_name);
            int w, h, ch;
            unsigned char *px = stbi_load(fp, &w, &h, &ch, 3);
            if (!px) continue;
            int pred;
            predict(px, w, h, NULL, &pred);
            stbi_image_free(px);
            cm[actual][pred]++;
            total++;
            if (actual == pred) correct++;
            cnt++;
        }
        closedir(d);
    }

    /* Tablo */
    printf("Confusion Matrix (gerçek \\ tahmin):\n\n");
    printf("             %10s %10s %10s %10s\n",
           CLASS_NAMES[0], CLASS_NAMES[1], CLASS_NAMES[2], CLASS_NAMES[3]);
    for (int r = 0; r < N_CLASSES; r++) {
        printf("%-13s %10d %10d %10d %10d\n",
               CLASS_NAMES[r], cm[r][0], cm[r][1], cm[r][2], cm[r][3]);
    }

    printf("\nPer-class recall:\n");
    for (int c = 0; c < N_CLASSES; c++) {
        int row_sum = cm[c][0] + cm[c][1] + cm[c][2] + cm[c][3];
        double recall = row_sum > 0 ? 100.0 * cm[c][c] / row_sum : 0.0;
        printf("  %-16s %6.2f%% (%d/%d)\n", CLASS_NAMES[c], recall, cm[c][c], row_sum);
    }

    printf("\nPer-class precision (tahmin edilenler içinde doğru):\n");
    for (int c = 0; c < N_CLASSES; c++) {
        int col_sum = cm[0][c] + cm[1][c] + cm[2][c] + cm[3][c];
        double prec = col_sum > 0 ? 100.0 * cm[c][c] / col_sum : 0.0;
        printf("  %-16s %6.2f%% (%d/%d)\n", CLASS_NAMES[c], prec, cm[c][c], col_sum);
    }

    printf("\nOVERALL: %.2f%% (%d/%d)\n", 100.0 * correct / total, correct, total);

    /* En büyük karışıklıklar */
    printf("\nTop confusions (non-diagonal):\n");
    int found = 0;
    for (int r = 0; r < N_CLASSES && found < 4; r++)
        for (int c = 0; c < N_CLASSES; c++)
            if (r != c && cm[r][c] > 0) {
                printf("  %-16s → %-16s : %d\n",
                       CLASS_NAMES[r], CLASS_NAMES[c], cm[r][c]);
                found++;
            }

    free_model();
    return 0;
}
