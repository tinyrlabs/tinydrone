/**
 * detector.h - Color-based object detection pipeline
 *
 * Pure C11, zero dependencies. Designed for both desktop simulation
 * and ESP32-CAM firmware (with CML_NO_MATH on embedded).
 */

#ifndef TINYDRONE_DETECTOR_H
#define TINYDRONE_DETECTOR_H

#include <stdint.h>
#include <stddef.h>

/* Image dimensions (processing resolution) */
#define TD_FRAME_W  320
#define TD_FRAME_H  240
#define TD_DOWNSAMPLE_W  32
#define TD_DOWNSAMPLE_H  32

/* Target color ranges in HSV */
typedef struct {
    uint8_t h_min, h_max;   /* Hue: 0-179 (OpenCV-style, H/2) */
    uint8_t s_min, s_max;   /* Saturation: 0-255 */
    uint8_t v_min, v_max;   /* Value: 0-255 */
} ColorRange;

/* Detection result */
typedef struct {
    int detected;            /* 1 if target found, 0 otherwise */
    int cx, cy;              /* Center of detected object (in downsample coords) */
    int x, y, w, h;          /* Bounding box (in downsample coords) */
    int area;                /* Pixel count of detected region */
    float confidence;        /* Simple confidence score (0-1) */
} Detection;

/* RGB pixel */
typedef struct {
    uint8_t r, g, b;
} RGB;

/* HSV pixel */
typedef struct {
    uint8_t h;    /* 0-179 */
    uint8_t s;    /* 0-255 */
    uint8_t v;    /* 0-255 */
} HSV;

/* ============================================
 * Color conversion
 * ============================================ */

/** RGB (0-255 each) to HSV (H:0-179, S:0-255, V:0-255) */
HSV rgb_to_hsv(RGB c);

/** Downsample a full-res RGB image to processing resolution.
 *  Uses simple averaging (box filter). */
void downsample_rgb(const RGB *src, int src_w, int src_h,
                    RGB *dst, int dst_w, int dst_h);

/* ============================================
 * Detection pipeline
 * ============================================ */

/**
 * Run full detection pipeline on a downsampled RGB frame.
 *
 * @param frame   RGB image, dst_w × dst_h pixels, row-major
 * @param w       Image width
 * @param h       Image height
 * @param target  Color range to look for
 * @param result  Output detection
 */
void detector_process(const RGB *frame, int w, int h,
                      const ColorRange *target, Detection *result);

/**
 * Create a binary mask: 1 where pixel matches color range, 0 elsewhere.
 *
 * @param frame   RGB input
 * @param mask    Output mask (must be w*h bytes)
 * @param w, h    Dimensions
 * @param target  Color range
 */
void detector_color_mask(const RGB *frame, uint8_t *mask,
                         int w, int h, const ColorRange *target);

/**
 * Find connected components in a binary mask using 4-connected flood fill.
 * Returns the largest component's bounding box and center.
 *
 * @param mask    Binary mask (modified in-place!)
 * @param w, h    Dimensions
 * @param result  Filled with largest component info
 */
void detector_find_largest(const uint8_t *mask, int w, int h,
                           Detection *result);

/**
 * Simple morphological open (erode then dilate) with 3x3 cross kernel.
 * Removes salt-and-pepper noise.
 *
 * @param mask    Binary mask (modified in-place)
 * @param w, h    Dimensions
 */
void detector_morph_open(uint8_t *mask, int w, int h);

/* ============================================
 * Tracking
 * ============================================ */

typedef struct {
    float kp;                /* Proportional gain */
    int deadzone;            /* Pixel deadzone around center */
    int pan_angle;           /* Current pan servo angle (degrees) */
    int tilt_angle;          /* Current tilt servo angle */
    int pan_min, pan_max;    /* Mechanical limits */
    int tilt_min, tilt_max;
    int target_lost_frames;  /* Consecutive frames without detection */
    int target_lost_timeout; /* Frames before declaring target lost */
} Tracker;

/** Initialize tracker with default parameters */
void tracker_init(Tracker *t);

/**
 * Update tracker based on detection result.
 * Returns required servo movements (delta angles).
 *
 * @param t      Tracker state
 * @param det    Current detection
 * @param dpan   Output: pan angle delta
 * @param dtilt  Output: tilt angle delta
 */
void tracker_update(Tracker *t, const Detection *det,
                    int *dpan, int *dtilt);

#endif /* TINYDRONE_DETECTOR_H */
