/**
 * debug_i8.c — tek görselde float vs int8 katman karşılaştırma (bağımsız)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "matrix.h"
#include "conv2d.h"
#include "pool2d.h"
#include "tinydrone_model.h"
#include "tinydrone_model_int8.h"

#define IMG_H 32
#define IMG_W 32
#define INPUT_SCALE (2.0 / 255.0)

static void relu_inplace(Matrix *m) {
    for (size_t i = 0; i < m->rows * m->cols; i++)
        if (m->data[i] < 0.0) m->data[i] = 0.0;
}

/* int8 conv1 (inference_int8.c'deki ile aynı mantık) */
static void conv1_i8(const int8_t *in, int8_t *out, float out_scale) {
    const int k = 3, pad = 1, in_c = 3, out_c = 16;
    for (int oc = 0; oc < out_c; oc++) {
        const int8_t *wrow = &c1_w_i8[oc * in_c * k * k];
        float ws = c1_w_scale[oc];
        for (int oy = 0; oy < 32; oy++) {
            for (int ox = 0; ox < 32; ox++) {
                int32_t acc = 0;
                for (int ic = 0; ic < in_c; ic++) {
                    for (int ky = 0; ky < k; ky++) {
                        int iy = oy + ky - pad;
                        if (iy < 0 || iy >= 32) continue;
                        for (int kx = 0; kx < k; kx++) {
                            int ix = ox + kx - pad;
                            if (ix < 0 || ix >= 32) continue;
                            int32_t iv = (int32_t)in[ic * 32 * 32 + iy * 32 + ix];
                            int32_t wv = (int32_t)wrow[ic * k * k + ky * k + kx];
                            acc += iv * wv;
                        }
                    }
                }
                float val = (float)acc * ws * INPUT_SCALE + (float)c1_b[oc];
                int32_t q = (int32_t)lroundf(val / out_scale);
                if (q > 127) q = 127;
                if (q < -128) q = -128;
                out[oc * 32 * 32 + oy * 32 + ox] = (int8_t)q;
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s image.png\n", argv[0]); return 1; }

    int w, h, ch;
    unsigned char *px = stbi_load(argv[1], &w, &h, &ch, 3);
    if (!px) return 1;

    double frame[32 * 32 * 3];
    for (int y = 0; y < 32; y++)
        for (int x = 0; x < 32; x++)
            for (int cc = 0; cc < 3; cc++)
                frame[cc * 1024 + y * 32 + x] = (double)px[(y * w + x) * 3 + cc] / 127.5 - 1.0;
    stbi_image_free(px);

    /* FLOAT conv1 */
    Conv2D *c1 = conv2d_create_with_weights(3, 16, 3, 3, 1, 1, 1, 1, c1_w, c1_b);
    Matrix *X = matrix_alloc(1, 3072);
    memcpy(X->data, frame, 3072 * sizeof(double));
    Matrix *f_a1 = conv2d_forward(c1, X, 1, 32, 32);

    float f_mn = 1e30f, f_mx = -1e30f;
    for (int i = 0; i < 16 * 32 * 32; i++) {
        if (f_a1->data[i] < f_mn) f_mn = f_a1->data[i];
        if (f_a1->data[i] > f_mx) f_mx = f_a1->data[i];
    }
    printf("FLOAT conv1: [%.4f, %.4f]\n", f_mn, f_mx);

    /* INT8 conv1 */
    int8_t Xi[3072];
    for (int i = 0; i < 3072; i++) {
        int32_t q = (int32_t)lroundf((float)frame[i] / INPUT_SCALE);
        if (q > 127) q = 127; if (q < -128) q = -128;
        Xi[i] = (int8_t)q;
    }
    /* Simetrik aktivasyon scale: kalibrasyondan */
    float a_scale_conv1 = fmaxf(fabsf(act_conv1_min), fabsf(act_conv1_max)) / 127.0f;
    int8_t i_c1[16 * 32 * 32];
    conv1_i8(Xi, i_c1, a_scale_conv1);

    float i_mn = 1e30f, i_mx = -1e30f;
    for (int i = 0; i < 16 * 32 * 32; i++) {
        float v = (float)i_c1[i] * a_scale_conv1;
        if (v < i_mn) i_mn = v;
        if (v > i_mx) i_mx = v;
    }
    printf("INT8  conv1 (dequant): [%.4f, %.4f] (scale=%.5f)\n", i_mn, i_mx, a_scale_conv1);

    /* İlk 5 kanalın ilk 3x3 çıktısını karşılaştır */
    printf("\nKanal bazlı örnek (oy=16, ox=16 civarı):\n");
    for (int oc = 0; oc < 5; oc++) {
        double fv = f_a1->data[oc * 32 * 32 + 16 * 32 + 16];
        double iv = (double)i_c1[oc * 32 * 32 + 16 * 32 + 16] * a_scale_conv1;
        printf("  ch%d: float=%.4f int8=%.4f fark=%.4f\n", oc, fv, iv, fv - iv);
    }

    /* ---------- TAM PIPELINE: float vs int8 (elle) ---------- */
    /* Float: conv1→relu→pool→conv2→relu→pool→fc1→relu→fc2 */
    Conv2D *c2 = conv2d_create_with_weights(16, 32, 3, 3, 1, 1, 1, 1, c2_w, c2_b);
    MaxPool2D *p1 = maxpool2d_create(2, 2, 2, 2);
    MaxPool2D *p2 = maxpool2d_create(2, 2, 2, 2);

    Matrix *f_a1r = matrix_copy(f_a1); relu_inplace(f_a1r);
    Matrix *f_p1 = maxpool2d_forward(p1, f_a1r, 1, 16, 32, 32);
    Matrix *f_a2 = conv2d_forward(c2, f_p1, 1, 16, 16);
    Matrix *f_a2r = matrix_copy(f_a2); relu_inplace(f_a2r);
    Matrix *f_p2 = maxpool2d_forward(p2, f_a2r, 1, 32, 16, 16);

    /* float FC1 */
    Matrix *f_fc1w = matrix_alloc(2048, 64);
    memcpy(f_fc1w->data, fc1_w, 2048 * 64 * sizeof(double));
    Matrix *f_fc1 = matrix_matmul(f_p2, f_fc1w);
    for (int c = 0; c < 64; c++) f_fc1->data[c] += fc1_b[c];
    relu_inplace(f_fc1);
    /* float FC2 */
    Matrix *f_fc2w = matrix_alloc(64, 4);
    memcpy(f_fc2w->data, fc2_w, 64 * 4 * sizeof(double));
    Matrix *f_out = matrix_matmul(f_fc1, f_fc2w);
    for (int c = 0; c < 4; c++) f_out->data[c] += fc2_b[c];

    printf("\nFLOAT logits: [%.3f %.3f %.3f %.3f]\n",
           f_out->data[0], f_out->data[1], f_out->data[2], f_out->data[3]);

    /* INT8 pipeline (elle) */
    int8_t c1o[16 * 32 * 32];
    conv1_i8(Xi, c1o, a_scale_conv1);
    /* relu */
    for (int i = 0; i < 16 * 32 * 32; i++) if (c1o[i] < 0) c1o[i] = 0;
    /* pool1 */
    int8_t p1o[16 * 16 * 16];
    for (int ic = 0; ic < 16; ic++)
        for (int oy = 0; oy < 16; oy++)
            for (int ox = 0; ox < 16; ox++) {
                int8_t m = c1o[ic * 1024 + (2 * oy) * 32 + (2 * ox)];
                int8_t v;
                v = c1o[ic * 1024 + (2 * oy) * 32 + (2 * ox + 1)]; if (v > m) m = v;
                v = c1o[ic * 1024 + (2 * oy + 1) * 32 + (2 * ox)]; if (v > m) m = v;
                v = c1o[ic * 1024 + (2 * oy + 1) * 32 + (2 * ox + 1)]; if (v > m) m = v;
                p1o[ic * 256 + oy * 16 + ox] = m;
            }
    /* conv2 */
    float a_scale_conv2 = fmaxf(fabsf(act_conv2_min), fabsf(act_conv2_max)) / 127.0f;
    int8_t c2o[32 * 16 * 16];
    const int k2 = 3, pad2 = 1;
    for (int oc = 0; oc < 32; oc++) {
        const int8_t *wrow = &c2_w_i8[oc * 16 * k2 * k2];
        float ws = c2_w_scale[oc];
        for (int oy = 0; oy < 16; oy++) {
            for (int ox = 0; ox < 16; ox++) {
                int32_t acc = 0;
                for (int ic = 0; ic < 16; ic++) {
                    for (int ky = 0; ky < k2; ky++) {
                        int iy = oy + ky - pad2;
                        if (iy < 0 || iy >= 16) continue;
                        for (int kx = 0; kx < k2; kx++) {
                            int ix = ox + kx - pad2;
                            if (ix < 0 || ix >= 16) continue;
                            int32_t iv = (int32_t)p1o[ic * 256 + iy * 16 + ix];
                            int32_t wv = (int32_t)wrow[ic * k2 * k2 + ky * k2 + kx];
                            acc += iv * wv;
                        }
                    }
                }
                float val = (float)acc * ws * a_scale_conv1 + (float)c2_b[oc];
                int32_t q = (int32_t)lroundf(val / a_scale_conv2);
                if (q > 127) q = 127; if (q < -128) q = -128;
                c2o[oc * 256 + oy * 16 + ox] = (int8_t)q;
            }
        }
    }
    /* relu2 */
    for (int i = 0; i < 32 * 16 * 16; i++) if (c2o[i] < 0) c2o[i] = 0;
    /* pool2 */
    int8_t p2o[32 * 8 * 8];
    for (int ic = 0; ic < 32; ic++)
        for (int oy = 0; oy < 8; oy++)
            for (int ox = 0; ox < 8; ox++) {
                int8_t m = c2o[ic * 256 + (2 * oy) * 16 + (2 * ox)];
                int8_t v;
                v = c2o[ic * 256 + (2 * oy) * 16 + (2 * ox + 1)]; if (v > m) m = v;
                v = c2o[ic * 256 + (2 * oy + 1) * 16 + (2 * ox)]; if (v > m) m = v;
                v = c2o[ic * 256 + (2 * oy + 1) * 16 + (2 * ox + 1)]; if (v > m) m = v;
                p2o[ic * 64 + oy * 8 + ox] = m;
            }
    /* fc1 int8 */
    float a_scale_fc1 = fmaxf(fabsf(act_fc1_min), fabsf(act_fc1_max)) / 127.0f;
    int8_t fc1o[64];
    for (int oc = 0; oc < 64; oc++) {
        /* input-major: w[i*64+oc] */
        int32_t acc = 0;
        for (int i = 0; i < 2048; i++)
            acc += (int32_t)p2o[i] * (int32_t)fc1_w_i8[i * 64 + oc];
        float val = (float)acc * (fc1_w_scale[oc] * a_scale_conv2) + (float)fc1_b[oc];
        int32_t q = (int32_t)lroundf(val / a_scale_fc1);
        if (q > 127) q = 127; if (q < -128) q = -128;
        fc1o[oc] = (int8_t)q;
    }
    /* relu fc1 */
    for (int i = 0; i < 64; i++) if (fc1o[i] < 0) fc1o[i] = 0;
    /* fc2 int8 */
    float logits[4];
    for (int oc = 0; oc < 4; oc++) {
        /* input-major: w[i*4+oc] */
        int32_t acc = 0;
        for (int i = 0; i < 64; i++)
            acc += (int32_t)fc1o[i] * (int32_t)fc2_w_i8[i * 4 + oc];
        logits[oc] = (float)acc * (fc2_w_scale[oc] * a_scale_fc1) + (float)fc2_b[oc];
    }
    printf("INT8  logits: [%.3f %.3f %.3f %.3f]\n", logits[0], logits[1], logits[2], logits[3]);

    /* İlk 8 FC1 nöronunu karşılaştır */
    printf("\nFC1 nöron karşılaştırma:\n");
    for (int oc = 0; oc < 8; oc++) {
        double fv = f_fc1->data[oc];
        double iv = (double)fc1o[oc] * a_scale_fc1;
        printf("  n%d: float=%.4f int8=%.4f fark=%.4f\n", oc, fv, iv, fv - iv);
    }

    matrix_free(X); matrix_free(f_a1); matrix_free(f_a1r); matrix_free(f_p1);
    matrix_free(f_a2); matrix_free(f_a2r); matrix_free(f_p2);
    matrix_free(f_fc1w); matrix_free(f_fc1);
    matrix_free(f_fc2w); matrix_free(f_out);
    conv2d_free(c1); conv2d_free(c2); maxpool2d_free(p1); maxpool2d_free(p2);
    return 0;
}
