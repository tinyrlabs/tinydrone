/**
 * display.h - ASCII/terminal visualization for tinydrone simulation
 *
 * Pure C11, zero dependencies. Outputs colored blocks to terminal.
 */

#ifndef TINYDRONE_DISPLAY_H
#define TINYDRONE_DISPLAY_H

#include "detector.h"

/** Initialize terminal (hide cursor, clear screen) */
void display_init(void);

/** Restore terminal on exit */
void display_cleanup(void);

/**
 * Render a frame with detection overlay to terminal.
 *
 * @param frame      RGB image
 * @param w, h       Dimensions
 * @param det        Detection result (can be NULL)
 * @param fps        Current FPS (0 to hide)
 * @param pan, tilt  Current servo angles
 */
void display_render(const RGB *frame, int w, int h,
                    const Detection *det, float fps,
                    int pan, int tilt);

/**
 * Render the binary mask (debug view).
 */
void display_render_mask(const uint8_t *mask, int w, int h,
                         const Detection *det);

#endif /* TINYDRONE_DISPLAY_H */
