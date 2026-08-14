/**
 * sliding_window.c — coarse-to-fine hedef tespiti
 */

#include "sliding_window.h"
#include <math.h>
#include <string.h>

/* Patch çıkar: frame (RGB888) → 32x32x3 double NCHW [-1,1] */
static void extract_patch(const uint8_t *frame, int fx, int fy,
                          double *patch) {
    for (int y = 0; y < SW_WIN; y++) {
        for (int x = 0; x < SW_WIN; x++) {
            const uint8_t *px = frame + ((fy + y) * SW_FRAME_W + (fx + x)) * 3;
            patch[0 * 1024 + y * 32 + x] = (double)px[0] / 127.5 - 1.0;
            patch[1 * 1024 + y * 32 + x] = (double)px[1] / 127.5 - 1.0;
            patch[2 * 1024 + y * 32 + x] = (double)px[2] / 127.5 - 1.0;
        }
    }
}

/* Coarse tarama: stride 16 → 9x7 = 63 konum */
static void scan_coarse(const uint8_t *frame,
                        int (*infer_cb)(const double *, double *),
                        int *best_x, int *best_y, double *best_conf,
                        int *best_cls, double patch_buf[3072]) {
    *best_conf = 0.0;
    *best_x = 0; *best_y = 0; *best_cls = 0;

    for (int fy = 0; fy + SW_WIN <= SW_FRAME_H; fy += SW_STRIDE_COARSE) {
        for (int fx = 0; fx + SW_WIN <= SW_FRAME_W; fx += SW_STRIDE_COARSE) {
            extract_patch(frame, fx, fy, patch_buf);
            double probs[SW_NUM_CLASSES];
            int pred = infer_cb(patch_buf, probs);
            if (pred != SW_TARGET_BG && probs[pred] > *best_conf) {
                *best_conf = probs[pred];
                *best_x = fx;
                *best_y = fy;
                *best_cls = pred;
            }
        }
    }
}

/* Fine tarama: best konum etrafında stride 4 (3x3 = 9 konum) */
static void scan_fine(const uint8_t *frame,
                      int (*infer_cb)(const double *, double *),
                      int cx, int cy, double *best_conf,
                      int *best_x, int *best_y, int *best_cls,
                      double patch_buf[3072]) {
    for (int dy = -SW_STRIDE_FINE; dy <= SW_STRIDE_FINE; dy += SW_STRIDE_FINE) {
        for (int dx = -SW_STRIDE_FINE; dx <= SW_STRIDE_FINE; dx += SW_STRIDE_FINE) {
            int fx = cx + dx, fy = cy + dy;
            if (fx < 0 || fy < 0 || fx + SW_WIN > SW_FRAME_W || fy + SW_WIN > SW_FRAME_H)
                continue;
            extract_patch(frame, fx, fy, patch_buf);
            double probs[SW_NUM_CLASSES];
            int pred = infer_cb(patch_buf, probs);
            if (pred != SW_TARGET_BG && probs[pred] > *best_conf) {
                *best_conf = probs[pred];
                *best_x = fx;
                *best_y = fy;
                *best_cls = pred;
            }
        }
    }
}

void sw_detect(const uint8_t *frame,
               int (*infer_cb)(const double *, double *),
               SWDetection *out) {
    double patch[32 * 32 * 3];

    out->detected = 0;
    out->confidence = 0.0;

    /* Coarse */
    int cx, cy, cls;
    double conf;
    scan_coarse(frame, infer_cb, &cx, &cy, &conf, &cls, patch);

    if (conf > 0.0) {
        /* Fine — coarse konum etrafında iyileştir */
        int fx = cx, fy = cy;
        scan_fine(frame, infer_cb, cx, cy, &conf, &fx, &fy, &cls, patch);
    }

    if (conf > 0.0) {
        out->detected = 1;
        out->x = cx;
        out->y = cy;
        out->w = SW_WIN;
        out->h = SW_WIN;
        out->cls = cls;
        out->confidence = conf;

        /* Bounding box merkezi → frame merkezine göre normalize offset */
        int bbox_cx = cx + SW_WIN / 2;
        int bbox_cy = cy + SW_WIN / 2;
        out->offset_x = (2.0 * bbox_cx / SW_FRAME_W) - 1.0;
        out->offset_y = (2.0 * bbox_cy / SW_FRAME_H) - 1.0;
    }
}

/* ── Takip modu (Faz 4.6) ───────────────────────────────────── */

void sw_tracker_init(SWTracker *t) {
    t->locked = 0;
    t->last_x = 0;
    t->last_y = 0;
    t->last_cls = 0;
    t->lost_count = 0;
}

/* Tek konumda inference: patch çıkar → tahmin + güven. */
static double infer_at(const uint8_t *frame, int fx, int fy,
                       int (*infer_cb)(const double *, double *),
                       double patch_buf[3072], int *cls_out) {
    extract_patch(frame, fx, fy, patch_buf);
    double probs[SW_NUM_CLASSES];
    int pred = infer_cb(patch_buf, probs);
    *cls_out = pred;
    return pred != SW_TARGET_BG ? probs[pred] : 0.0;
}

/* Kilitli konum etrafında 3x3 arama (stride 4). */
static double search_around(const uint8_t *frame,
                            int (*infer_cb)(const double *, double *),
                            int cx, int cy,
                            int *best_x, int *best_y, int *best_cls,
                            double patch_buf[3072]) {
    double best = 0.0;
    *best_x = cx;
    *best_y = cy;
    *best_cls = 0;

    for (int dy = -SW_STRIDE_FINE; dy <= SW_STRIDE_FINE; dy += SW_STRIDE_FINE) {
        for (int dx = -SW_STRIDE_FINE; dx <= SW_STRIDE_FINE; dx += SW_STRIDE_FINE) {
            int fx = cx + dx, fy = cy + dy;
            if (fx < 0 || fy < 0 || fx + SW_WIN > SW_FRAME_W || fy + SW_WIN > SW_FRAME_H)
                continue;
            int cls;
            double c = infer_at(frame, fx, fy, infer_cb, patch_buf, &cls);
            if (c > best) {
                best = c;
                *best_x = fx;
                *best_y = fy;
                *best_cls = cls;
            }
        }
    }
    return best;
}

/* Sonucu doldur (bbox + offset). */
static void fill_detection(SWDetection *out, int x, int y, int cls, double conf) {
    out->detected = 1;
    out->x = x;
    out->y = y;
    out->w = SW_WIN;
    out->h = SW_WIN;
    out->cls = cls;
    out->confidence = conf;
    int bbox_cx = x + SW_WIN / 2;
    int bbox_cy = y + SW_WIN / 2;
    out->offset_x = (2.0 * bbox_cx / SW_FRAME_W) - 1.0;
    out->offset_y = (2.0 * bbox_cy / SW_FRAME_H) - 1.0;
}

void sw_detect_track(const uint8_t *frame,
                     int (*infer_cb)(const double *, double *),
                     SWTracker *t, SWDetection *out) {
    double patch[32 * 32 * 3];
    out->detected = 0;
    out->confidence = 0.0;

    if (!t->locked) {
        /* Tam tarama — hedef ara */
        SWDetection d;
        sw_detect(frame, infer_cb, &d);
        if (d.detected && d.confidence >= SW_LOCK_CONF) {
            t->locked = 1;
            t->last_x = d.x;
            t->last_y = d.y;
            t->last_cls = d.cls;
            t->lost_count = 0;
            *out = d;
        }
        return;
    }

    /* Kilitli: 1) önceki konumda tek inference */
    int cls;
    double conf = infer_at(frame, t->last_x, t->last_y, infer_cb, patch, &cls);

    if (conf >= SW_LOCK_CONF && cls == t->last_cls) {
        /* Hedef hâlâ orada, sınıf aynı — kilit devam */
        t->lost_count = 0;
        fill_detection(out, t->last_x, t->last_y, cls, conf);
        return;
    }

    if (conf >= SW_LOST_CONF) {
        /* Düşük güven veya sınıf değişimi — çevre ara */
        int bx, by, bcls;
        double bc = search_around(frame, infer_cb, t->last_x, t->last_y,
                                  &bx, &by, &bcls, patch);
        if (bc >= SW_LOCK_CONF && bcls == t->last_cls) {
            t->lost_count = 0;
            t->last_x = bx;
            t->last_y = by;
            fill_detection(out, bx, by, bcls, bc);
            return;
        }
        if (bc >= SW_LOST_CONF) {
            /* Orta güven veya sınıf farklı — konumu güncelle ama sınıfı
             * KORU ve sayacı artır (sınıf değişimi güvenli değilse kilit
             * 3 kare içinde çözülür, yeniden tarama başlar) */
            t->last_x = bx;
            t->last_y = by;
            fill_detection(out, bx, by, bcls, bc);
            t->lost_count++;
            if (t->lost_count >= SW_MAX_LOST) {
                t->locked = 0;
                t->lost_count = 0;
            }
            return;
        }
    }

    /* Kayıp kare */
    t->lost_count++;
    if (t->lost_count >= SW_MAX_LOST) {
        /* Kalıcı kayıp — tam taramaya dön */
        t->locked = 0;
        t->lost_count = 0;
    }
}
