/**
 * demo.c — terminal görsel demo: model tahmini + ASCII görüntü
 *
 * Test setinden rastgele görsel seçer, modelle sınıflandırır,
 * ASCII render + güven skorları gösterir.
 *
 * Build:
 *   cc -std=c11 -O2 -I/path/to/tinycml/include -I. -Ioutput \
 *      demo.c /path/to/tinycml/build/lib/libtinycml.a -lm -o demo
 *
 * Run:
 *   ./demo dataset/processed [adet]
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
#define N_CLASSES 5
#define NCHW (IMG_C * IMG_H * IMG_W)

static const char *CLASS_NAMES[N_CLASSES] = {
    "tank", "armored_vehicle", "drone_uav", "background", "military_building"
};

static Conv2D *c1, *c2;
static MaxPool2D *p1, *p2;
static Matrix *m_fc1_w, *m_fc1_b, *m_fc2_w, *m_fc2_b;

static void relu_inplace(Matrix *m) {
    for (size_t i = 0; i < m->rows * m->cols; i++)
        if (m->data[i] < 0.0) m->data[i] = 0.0;
}

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

static int predict(const unsigned char *px, int w, int h, double *probs_out) {
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
    matrix_free(X); matrix_free(out);
    return best;
}

/* 32x32 görüntüyü 64x32 ASCII blok olarak bas (2 char per pixel, gri ton) */
static void render_ascii(const unsigned char *px, int w, int h) {
    const char *ramp = " .:-=+*#%@";
    for (int y = 0; y < IMG_H; y += 2) {
        for (int x = 0; x < IMG_W; x++) {
            int src_idx = (y * w + x) * 3;
            int gray = (px[src_idx] + px[src_idx + 1] + px[src_idx + 2]) / 3;
            int idx = gray * 9 / 256;
            putchar(ramp[idx]);
            putchar(ramp[idx]);
        }
        putchar('\n');
    }
}

int main(int argc, char **argv) {
    const char *data_root = argc > 1 ? argv[1] : "dataset/processed";
    int count = argc > 2 ? atoi(argv[2]) : 5;

    srand((unsigned)time(NULL));
    build_model();

    printf("tinydrone — Canlı Demo (terminal)\n");
    printf("=================================\n");
    printf("Model: tinycml CNN — %d sınıf\n\n", N_CLASSES);

    /* Her sınıftan görsel listesi topla */
    typedef struct { char path[2048]; int label; } Entry;
    Entry *entries = NULL;
    int n = 0;

    for (int c = 0; c < N_CLASSES; c++) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/test/%s", data_root, CLASS_NAMES[c]);
        DIR *d = opendir(path);
        if (!d) continue;
        struct dirent *e;
        while ((e = readdir(d))) {
            if (!strstr(e->d_name, ".png")) continue;
            entries = realloc(entries, (size_t)(n + 1) * sizeof(Entry));
            snprintf(entries[n].path, sizeof(entries[n].path), "%s/%s", path, e->d_name);
            entries[n].label = c;
            n++;
        }
        closedir(d);
    }
    printf("%d test görseli yüklendi.\n\n", n);

    int correct = 0;
    for (int i = 0; i < count && n > 0; i++) {
        int idx = rand() % n;
        int w, h, ch;
        unsigned char *px = stbi_load(entries[idx].path, &w, &h, &ch, 3);
        if (!px) continue;

        double probs[N_CLASSES];
        int pred = predict(px, w, h, probs);

        printf("━━━ Görsel %d: %s (gerçek: %s) ━━━\n",
               i + 1, strrchr(entries[idx].path, '/') + 1, CLASS_NAMES[entries[idx].label]);
        render_ascii(px, w, h);

        printf("\n  Tahmin: %s %s\n", CLASS_NAMES[pred],
               pred == entries[idx].label ? "✅" : "❌");
        for (int c = 0; c < N_CLASSES; c++) {
            printf("    %-16s %6.1f%% %s\n", CLASS_NAMES[c], probs[c] * 100.0,
                   c == pred ? "◀" : "");
        }
        printf("\n");

        if (pred == entries[idx].label) correct++;
        stbi_image_free(px);
    }

    printf("═══════════════════════\n");
    printf("Demo doğruluğu: %d/%d\n", correct, count);

    free(entries);
    free_model();
    return 0;
}
