/**
 * detector.c - Color-based object detection implementation
 */

#include "detector.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

/* ============================================
 * Color conversion
 * ============================================ */

HSV rgb_to_hsv(RGB c) {
    HSV out;
    uint8_t r = c.r, g = c.g, b = c.b;

    uint8_t mx = r > g ? (r > b ? r : b) : (g > b ? g : b);
    uint8_t mn = r < g ? (r < b ? r : b) : (g < b ? g : b);
    uint8_t delta = mx - mn;

    /* Value */
    out.v = mx;

    /* Saturation */
    if (mx == 0) {
        out.s = 0;
    } else {
        out.s = (uint8_t)((int)delta * 255 / mx);
    }

    /* Hue (0-179, OpenCV-style H/2) */
    if (delta == 0) {
        out.h = 0;
    } else if (mx == r) {
        out.h = (uint8_t)(30 * ((int)(g - b) * 60 / delta + 360) / 360 % 180);
        /* Simplified: */
        int htemp = 60 * (g - b) / delta;
        if (htemp < 0) htemp += 360;
        out.h = (uint8_t)(htemp / 2);
    } else if (mx == g) {
        int htemp = 60 * (b - r) / delta + 120;
        out.h = (uint8_t)(htemp / 2);
    } else {
        int htemp = 60 * (r - g) / delta + 240;
        out.h = (uint8_t)(htemp / 2);
    }

    return out;
}

void downsample_rgb(const RGB *src, int src_w, int src_h,
                    RGB *dst, int dst_w, int dst_h) {
    float scale_x = (float)src_w / dst_w;
    float scale_y = (float)src_h / dst_h;

    for (int dy = 0; dy < dst_h; dy++) {
        for (int dx = 0; dx < dst_w; dx++) {
            /* Source region in original image */
            int sx_start = (int)(dx * scale_x);
            int sy_start = (int)(dy * scale_y);
            int sx_end   = (int)((dx + 1) * scale_x);
            int sy_end   = (int)((dy + 1) * scale_y);
            if (sx_end > src_w) sx_end = src_w;
            if (sy_end > src_h) sy_end = src_h;
            if (sx_start >= sx_end) sx_end = sx_start + 1;
            if (sy_start >= sy_end) sy_end = sy_start + 1;

            /* Box average */
            int sum_r = 0, sum_g = 0, sum_b = 0;
            int count = 0;
            for (int sy = sy_start; sy < sy_end; sy++) {
                for (int sx = sx_start; sx < sx_end; sx++) {
                    RGB p = src[sy * src_w + sx];
                    sum_r += p.r;
                    sum_g += p.g;
                    sum_b += p.b;
                    count++;
                }
            }
            dst[dy * dst_w + dx].r = (uint8_t)(sum_r / count);
            dst[dy * dst_w + dx].g = (uint8_t)(sum_g / count);
            dst[dy * dst_w + dx].b = (uint8_t)(sum_b / count);
        }
    }
}

/* ============================================
 * Detection pipeline
 * ============================================ */

void detector_color_mask(const RGB *frame, uint8_t *mask,
                         int w, int h, const ColorRange *target) {
    for (int i = 0; i < w * h; i++) {
        HSV hsv = rgb_to_hsv(frame[i]);
        int match = (hsv.h >= target->h_min && hsv.h <= target->h_max &&
                     hsv.s >= target->s_min && hsv.s <= target->s_max &&
                     hsv.v >= target->v_min && hsv.v <= target->v_max);
        mask[i] = match ? 1 : 0;
    }
}

/* 3x3 cross kernel morph ops */
static void morph_erode(uint8_t *dst, const uint8_t *src, int w, int h) {
    memcpy(dst, src, (size_t)w * h);
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int idx = y * w + x;
            if (src[idx] == 0) continue;
            /* Require N, S, E, W, center all 1 */
            if (src[idx - w] && src[idx + w] &&
                src[idx - 1] && src[idx + 1]) {
                dst[idx] = 1;
            } else {
                dst[idx] = 0;
            }
        }
    }
}

static void morph_dilate(uint8_t *dst, const uint8_t *src, int w, int h) {
    memcpy(dst, src, (size_t)w * h);
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int idx = y * w + x;
            if (src[idx]) continue;
            /* If any of N, S, E, W is 1, set to 1 */
            if (src[idx - w] || src[idx + w] ||
                src[idx - 1] || src[idx + 1]) {
                dst[idx] = 1;
            }
        }
    }
}

void detector_morph_open(uint8_t *mask, int w, int h) {
    uint8_t *tmp = malloc((size_t)w * h);
    if (!tmp) return;
    morph_erode(tmp, mask, w, h);
    morph_dilate(mask, tmp, w, h);
    free(tmp);
}

/* Flood fill helper for connected component labeling */
typedef struct {
    int x, y;
} Point;

static int flood_fill(uint8_t *mask, int w, int h, int sx, int sy,
                      int *min_x, int *min_y, int *max_x, int *max_y) {
    /* Simple stack-based flood fill (4-connected) */
    Point *stack = malloc((size_t)w * h * sizeof(Point));
    if (!stack) return 0;

    int top = 0;
    int count = 0;

    stack[top].x = sx;
    stack[top].y = sy;
    top++;
    mask[sy * w + sx] = 0;  /* Mark as visited */

    *min_x = sx; *max_x = sx;
    *min_y = sy; *max_y = sy;

    while (top > 0) {
        top--;
        int x = stack[top].x;
        int y = stack[top].y;
        count++;

        if (x < *min_x) *min_x = x;
        if (x > *max_x) *max_x = x;
        if (y < *min_y) *min_y = y;
        if (y > *max_y) *max_y = y;

        /* 4-connected neighbors */
        int neighbors[4][2] = {{x-1,y}, {x+1,y}, {x,y-1}, {x,y+1}};
        for (int n = 0; n < 4; n++) {
            int nx = neighbors[n][0];
            int ny = neighbors[n][1];
            if (nx >= 0 && nx < w && ny >= 0 && ny < h && mask[ny * w + nx]) {
                mask[ny * w + nx] = 0;  /* Mark visited */
                stack[top].x = nx;
                stack[top].y = ny;
                top++;
            }
        }
    }

    free(stack);
    return count;
}

void detector_find_largest(const uint8_t *mask, int w, int h,
                           Detection *result) {
    memset(result, 0, sizeof(*result));

    /* Make a copy since flood fill modifies the mask */
    uint8_t *work = malloc((size_t)w * h);
    if (!work) return;
    memcpy(work, mask, (size_t)w * h);

    int best_area = 0;
    int best_cx = 0, best_cy = 0;
    int best_x = 0, best_y = 0, best_w = 0, best_h = 0;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            if (work[y * w + x]) {
                int min_x, min_y, max_x, max_y;
                int area = flood_fill(work, w, h, x, y,
                                      &min_x, &min_y, &max_x, &max_y);
                if (area > best_area) {
                    best_area = area;
                    best_cx = (min_x + max_x) / 2;
                    best_cy = (min_y + max_y) / 2;
                    best_x = min_x;
                    best_y = min_y;
                    best_w = max_x - min_x + 1;
                    best_h = max_y - min_y + 1;
                }
            }
        }
    }

    free(work);

    if (best_area > 4) {  /* Minimum area threshold */
        result->detected = 1;
        result->cx = best_cx;
        result->cy = best_cy;
        result->x = best_x;
        result->y = best_y;
        result->w = best_w;
        result->h = best_h;
        result->area = best_area;

        /* Simple confidence: ratio of area to bounding box */
        int bb_area = best_w * best_h;
        result->confidence = (bb_area > 0) ? (float)best_area / bb_area : 0.0f;
    }
}

void detector_process(const RGB *frame, int w, int h,
                      const ColorRange *target, Detection *result) {
    uint8_t *mask = malloc((size_t)w * h);
    if (!mask) {
        memset(result, 0, sizeof(*result));
        return;
    }

    /* 1. Color mask */
    detector_color_mask(frame, mask, w, h, target);

    /* 2. Morphological opening (remove noise) */
    detector_morph_open(mask, w, h);

    /* 3. Find largest connected component */
    detector_find_largest(mask, w, h, result);

    free(mask);
}

/* ============================================
 * Tracking
 * ============================================ */

void tracker_init(Tracker *t) {
    memset(t, 0, sizeof(*t));
    t->kp = 0.5f;
    t->deadzone = 2;
    t->pan_angle = 90;
    t->tilt_angle = 90;
    t->pan_min = 0;
    t->pan_max = 180;
    t->tilt_min = 30;
    t->tilt_max = 150;
    t->target_lost_timeout = 30;  /* ~1 second at 30fps */
}

void tracker_update(Tracker *t, const Detection *det,
                    int *dpan, int *dtilt) {
    *dpan = 0;
    *dtilt = 0;

    if (!det->detected) {
        t->target_lost_frames++;
        return;
    }

    t->target_lost_frames = 0;

    /* Center of frame (assuming w/2, h/2 — caller passes frame dims) */
    int frame_cx = TD_DOWNSAMPLE_W / 2;
    int frame_cy = TD_DOWNSAMPLE_H / 2;

    int dx = det->cx - frame_cx;
    int dy = det->cy - frame_cy;

    /* Deadzone check */
    if (abs(dx) > t->deadzone) {
        *dpan = (int)(-t->kp * dx);
    }
    if (abs(dy) > t->deadzone) {
        *dtilt = (int)(t->kp * dy);  /* Positive dy = object below center, tilt down */
    }

    /* Apply limits */
    int new_pan = t->pan_angle + *dpan;
    int new_tilt = t->tilt_angle + *dtilt;
    if (new_pan < t->pan_min) { *dpan = t->pan_min - t->pan_angle; }
    if (new_pan > t->pan_max) { *dpan = t->pan_max - t->pan_angle; }
    if (new_tilt < t->tilt_min) { *dtilt = t->tilt_min - t->tilt_angle; }
    if (new_tilt > t->tilt_max) { *dtilt = t->tilt_max - t->tilt_angle; }

    t->pan_angle += *dpan;
    t->tilt_angle += *dtilt;
}
