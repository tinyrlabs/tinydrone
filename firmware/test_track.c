/**
 * test_track.c — takip modu host testi
 *
 * Senaryo: hedefi bir konuma koy → tracker kilitle → hedefi yavaşça
 * hareket ettir → tracker takip etsin → hedefi kaldır → 3 kare sonra
 * tam taramaya dönsün. Inference sayısını sayaçla (FPS eşdeğeri).
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

static int g_inferences = 0;

static int infer_cb(const double *patch, double *probs) {
    g_inferences++;
    return inference_int8_run(patch, probs);
}

/* 160x120 frame oluştur: background + hedef görseli (px, py) konumunda */
static void make_frame(uint8_t *frame, const uint8_t *bg, int bw, int bh,
                       const uint8_t *tg, int tw, int th,
                       int px, int py) {
    for (int y = 0; y < SW_FRAME_H; y++)
        for (int x = 0; x < SW_FRAME_W; x++) {
            int sx = x * bw / SW_FRAME_W;
            int sy = y * bh / SW_FRAME_H;
            memcpy(&frame[(y * SW_FRAME_W + x) * 3], &bg[(sy * bw + sx) * 3], 3);
        }
    for (int y = 0; y < 32; y++) {
        int fy = py + y;
        if (fy < 0 || fy >= SW_FRAME_H) continue;
        for (int x = 0; x < 32; x++) {
            int fx = px + x;
            if (fx < 0 || fx >= SW_FRAME_W) continue;
            int sx = x * tw / 32;
            int sy = y * th / 32;
            memcpy(&frame[(fy * SW_FRAME_W + fx) * 3],
                   &tg[(sy * tw + sx) * 3], 3);
        }
    }
}

int main(void) {
    inference_int8_init();

    /* Görseller */
    int bw, bh, bc;
    unsigned char *bg = stbi_load("dataset/processed/test/background/other_other_0_100.png",
                                  &bw, &bh, &bc, 3);
    int tw, th, tc;
    unsigned char *tg = stbi_load("dataset/processed/test/drone_uav/Database1_100.png",
                                  &tw, &th, &tc, 3);
    if (!bg || !tg) { fprintf(stderr, "görsel yüklenemedi\n"); return 1; }

    uint8_t frame[SW_FRAME_W * SW_FRAME_H * 3];
    SWTracker trk;
    sw_tracker_init(&trk);
    SWDetection det;

    int pass = 0, fail = 0;

    /* 1) Hedefi (60,40)'a koy — kilit beklenir (tam tarama, ~72 inference) */
    g_inferences = 0;
    make_frame(frame, bg, bw, bh, tg, tw, th, 60, 40);
    sw_detect_track(frame, infer_cb, &trk, &det);
    printf("[1] İlk kare (tam tarama): inferences=%d kilit=%d detected=%d\n",
           g_inferences, trk.locked, det.detected);
    if (trk.locked && det.detected) { printf("    PASS ✓\n"); pass++; }
    else { printf("    FAIL ✗\n"); fail++; }

    /* 2) Kilitli — hedef sabit: 1 inference beklenir (hızlı mod) */
    g_inferences = 0;
    make_frame(frame, bg, bw, bh, tg, tw, th, 60, 40);
    sw_detect_track(frame, infer_cb, &trk, &det);
    printf("[2] Kilitli + sabit: inferences=%d (beklenen: 1) kilit=%d\n",
           g_inferences, trk.locked);
    if (g_inferences <= 3 && trk.locked) { printf("    PASS ✓\n"); pass++; }
    else { printf("    FAIL ✗\n"); fail++; }

    /* 3) Hedef 8px kaydı — çevre arama (≤10 inference) beklenir */
    g_inferences = 0;
    make_frame(frame, bg, bw, bh, tg, tw, th, 68, 40);
    sw_detect_track(frame, infer_cb, &trk, &det);
    printf("[3] Kilitli + kaymış (8px): inferences=%d kilit=%d bbox=(%d,%d)\n",
           g_inferences, trk.locked, det.x, det.y);
    if (trk.locked && g_inferences <= 12) { printf("    PASS ✓\n"); pass++; }
    else { printf("    FAIL ✗\n"); fail++; }

    /* 4) Hedef 12px daha kaydı */
    g_inferences = 0;
    make_frame(frame, bg, bw, bh, tg, tw, th, 80, 44);
    sw_detect_track(frame, infer_cb, &trk, &det);
    printf("[4] Kilitli + kaymış (12px): inferences=%d kilit=%d bbox=(%d,%d)\n",
           g_inferences, trk.locked, det.x, det.y);
    if (trk.locked) { printf("    PASS ✓\n"); pass++; }
    else { printf("    FAIL ✗\n"); fail++; }

    /* 5) Sınıf değişimi — drone kilitliyken yerine tank gelirse,
     * sınıf tutarlılığı kiliti çözmeli (3 kare içinde) */
    int unlock_frames = 0;
    for (int k = 0; k < 5; k++) {
        g_inferences = 0;
        make_frame(frame, bg, bw, bh, tg, tw, th, 60, 40);
        /* tank görseli ile değiştir */
        int tw2, th2, tc2;
        unsigned char *tg2 = stbi_load("dataset/processed/test/tank/tankds_00026.png",
                                       &tw2, &th2, &tc2, 3);
        for (int y = 0; y < 32; y++)
            for (int x = 0; x < 32; x++) {
                int sx = x * tw2 / 32, sy = y * th2 / 32;
                memcpy(&frame[((40 + y) * SW_FRAME_W + (60 + x)) * 3],
                       &tg2[(sy * tw2 + sx) * 3], 3);
            }
        stbi_image_free(tg2);
        sw_detect_track(frame, infer_cb, &trk, &det);
        printf("[5.%d] sınıf değişimi: inferences=%d kilit=%d detected=%d conf=%.2f cls=%d (beklenen cls=2)\n",
               k + 1, g_inferences, trk.locked, det.detected, det.confidence, det.cls);
        if (!trk.locked) { unlock_frames = k + 1; break; }
    }
    printf("[5] Kilit çözülme (sınıf değişimi): %d. karede (beklenen: ≤4) — %s\n",
           unlock_frames ? unlock_frames : 5,
           (unlock_frames > 0 && unlock_frames <= 4) ? "PASS ✓" : "FAIL ✗");
    if (unlock_frames > 0 && unlock_frames <= 4) pass++; else fail++;

    /* 6) Yeniden kilitleme — hedef geri gelirse tam tarama bulur */
    g_inferences = 0;
    make_frame(frame, bg, bw, bh, tg, tw, th, 30, 60);
    sw_detect_track(frame, infer_cb, &trk, &det);
    printf("[6] Yeniden kilitleme: inferences=%d kilit=%d\n", g_inferences, trk.locked);
    if (trk.locked) { printf("    PASS ✓\n"); pass++; }
    else { printf("    FAIL ✗\n"); fail++; }

    printf("\n================ SONUÇ: %d PASS, %d FAIL ================\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
