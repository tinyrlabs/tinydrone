/**
 * sliding_window.h — 160x120 kamerada 32x32 pencere ile hedef tespiti
 *
 * Coarse-to-fine: önce stride 16 ile tara, en yüksek güvenli bölgede
 * stride 4 ile detaylandır. Hedef bounding box + merkez offset döner.
 *
 * Hem host simülasyon hem firmware için (ESP-IDF'den bağımsız).
 */

#ifndef TINYDRONE_SLIDING_WINDOW_H
#define TINYDRONE_SLIDING_WINDOW_H

#include <stdint.h>

#define SW_FRAME_W 160
#define SW_FRAME_H 120
#define SW_WIN 32
#define SW_STRIDE_COARSE 16
#define SW_STRIDE_FINE 4
#define SW_NUM_CLASSES 4

/* Hedef sınıflar (background hariç): tank, armored, drone */
#define SW_TARGET_BG 3

typedef struct {
    int detected;        /* 1 = hedef bulundu */
    int x, y, w, h;      /* bounding box (frame koordinatları) */
    int cls;             /* hedef sınıf */
    double confidence;   /* max güven */
    double offset_x;     /* [-1, 1] — bbox merkezi frame merkezine göre */
    double offset_y;     /* [-1, 1] */
} SWDetection;

/**
 * Frame'de hedef ara.
 *
 * @param frame     RGB888 160x120 buffer (row-major, 3 byte/pixel)
 * @param infer_cb  int8 inference callback: patch (32x32x3 double NCHW) → (pred, probs)
 * @param out       sonuç
 */
void sw_detect(const uint8_t *frame,
               int (*infer_cb)(const double *patch, double *probs),
               SWDetection *out);

#endif /* TINYDRONE_SLIDING_WINDOW_H */
