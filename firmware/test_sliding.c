/**
 * test_sliding.c — sliding window host testi
 *
 * 160x120 frame oluştur: background + içine gerçek hedef görseli yerleştir.
 * Sliding window hedefi bulmalı.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "inference_int8.h"
#include "sliding_window.h"

/* int8 inference callback — patch (double NCHW) → pred/probs */
static int infer_cb(const double *patch, double *probs) {
    return inference_int8_run(patch, probs);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s background.png target.png [tx] [ty]\n", argv[0]);
        return 1;
    }
    const char *bg_path = argv[1];
    const char *tg_path = argv[2];
    int tx = argc > 3 ? atoi(argv[3]) : 60;
    int ty = argc > 4 ? atoi(argv[4]) : 40;

    inference_int8_init();
    printf("int8 model hazır\n");

    /* Background yükle ve 160x120'ye küçült */
    int bw, bh, bc;
    unsigned char *bg = stbi_load(bg_path, &bw, &bh, &bc, 3);
    if (!bg) { fprintf(stderr, "bg load fail\n"); return 1; }

    uint8_t frame[SW_FRAME_W * SW_FRAME_H * 3];
    for (int y = 0; y < SW_FRAME_H; y++)
        for (int x = 0; x < SW_FRAME_W; x++) {
            int sx = x * bw / SW_FRAME_W;
            int sy = y * bh / SW_FRAME_H;
            memcpy(&frame[(y * SW_FRAME_W + x) * 3], &bg[(sy * bw + sx) * 3], 3);
        }
    stbi_image_free(bg);
    printf("Frame hazır (160x120)\n");

    /* Hedef görseli yerleştir (32x32'ye küçült) */
    int tw, th, tc;
    unsigned char *tg = stbi_load(tg_path, &tw, &th, &tc, 3);
    if (!tg) { fprintf(stderr, "target load fail\n"); return 1; }
    for (int y = 0; y < 32; y++)
        for (int x = 0; x < 32; x++) {
            int sx = x * tw / 32;
            int sy = y * th / 32;
            memcpy(&frame[((ty + y) * SW_FRAME_W + (tx + x)) * 3],
                   &tg[(sy * tw + sx) * 3], 3);
        }
    stbi_image_free(tg);
    printf("Hedef (%s) yerleştirildi: (%d, %d)\n\n", tg_path, tx, ty);

    /* Tespit */
    SWDetection det;
    double t0 = (double)clock() / CLOCKS_PER_SEC;
    sw_detect(frame, infer_cb, &det);
    double elapsed = (double)clock() / CLOCKS_PER_SEC - t0;

    if (det.detected) {
        printf("TESPİT: sınıf=%d güven=%.2f\n", det.cls, det.confidence);
        printf("  bbox: (%d, %d) %dx%d\n", det.x, det.y, det.w, det.h);
        printf("  offset: x=%.2f y=%.2f\n", det.offset_x, det.offset_y);

        /* Doğruluk: bbox merkezi hedef merkezine yakın mı? */
        int true_cx = tx + 16, true_cy = ty + 16;
        int det_cx = det.x + 16, det_cy = det.y + 16;
        int err = abs(det_cx - true_cx) + abs(det_cy - true_cy);
        printf("  merkez hatası: %d px (hedef %d,%d → tespit %d,%d)\n",
               err, true_cx, true_cy, det_cx, det_cy);
        printf("  sonuç: %s\n", err < 16 ? "PASS ✓" : "FAIL ✗");
    } else {
        printf("Hedef BULUNAMADI ✗\n");
    }
    printf("  süre: %.0f ms (coarse+fine)\n", elapsed * 1000);

    return det.detected ? 0 : 1;
}
