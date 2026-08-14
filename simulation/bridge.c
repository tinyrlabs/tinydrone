/**
 * bridge.c — tinycml CNN köprüsü (simülasyon için)
 *
 * ESP32 firmware'deki AYNI kod (inference_int8.c + sliding_window.c) paylaşımlı
 * kütüphaneye derlenir ve Python (ctypes) ile çağrılır. Böylece simülasyonda
 * gömülüde çalışan kodun birebir aynısı çalışır.
 *
 * API (ctypes uyumlu):
 *   td_init()                       — model + tracker başlat
 *   td_detect_track(frame, w, h,    — 160x120 RGB frame → tespit/takip
 *                   &x, &y, &cls, &conf, &locked)
 *   td_track_state(&locked, &lost, &last_x, &last_y, &last_cls)
 */

#include <stdint.h>
#include <string.h>
#include "sliding_window.h"
#include "inference_int8.h"

static SWTracker g_trk;
static int g_inited = 0;

/* frame 160x120x3 (RGB) — sliding window API'si uint8_t* bekliyor, aynı */
static int infer_cb(const double *patch, double *probs) {
    return inference_int8_run(patch, probs);
}

void td_init(void) {
    inference_int8_init();
    sw_tracker_init(&g_trk);
    g_inited = 1;
}

int td_inited(void) {
    return g_inited;
}

/* Dönüş: 1 = tespit var, 0 = yok. Çıktılar işaretçiyle döner. */
int td_detect_track(const uint8_t *frame, int w, int h,
                    int *out_x, int *out_y, int *out_cls,
                    double *out_conf, int *out_locked) {
    if (!g_inited || !frame || w != SW_FRAME_W || h != SW_FRAME_H)
        return -1;
    SWDetection det;
    sw_detect_track(frame, infer_cb, &g_trk, &det);
    if (out_x) *out_x = det.detected ? det.x : -1;
    if (out_y) *out_y = det.detected ? det.y : -1;
    if (out_cls) *out_cls = det.detected ? det.cls : -1;
    if (out_conf) *out_conf = det.confidence;
    if (out_locked) *out_locked = g_trk.locked;
    return det.detected;
}

void td_track_state(int *locked, int *lost, int *last_x, int *last_y, int *last_cls) {
    if (locked) *locked = g_trk.locked;
    if (lost) *lost = g_trk.lost_count;
    if (last_x) *last_x = g_trk.last_x;
    if (last_y) *last_y = g_trk.last_y;
    if (last_cls) *last_cls = g_trk.last_cls;
}
