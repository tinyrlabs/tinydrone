/**
 * display.c - ASCII terminal visualization
 *
 * Uses ANSI escape codes for colors and cursor movement.
 * Each pixel is rendered as 2 characters (to approximate square aspect ratio).
 * Dark pixels: "  " (space), bright pixels: "██" (block).
 */

#include "display.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* ANSI escape sequences */
#define CSI     "\x1b["
#define CURSOR_HOME   CSI "H"
#define CLEAR_SCREEN  CSI "2J"
#define HIDE_CURSOR   CSI "?25l"
#define SHOW_CURSOR   CSI "?25h"
#define RESET         CSI "0m"

/* 8-color palette mapping from RGB — simple threshold approach */
static const char* rgb_to_ansi_bg(RGB c) {
    /* Simple brightness threshold approach for 8-color terminal */
    int bright = (int)c.r + c.g + c.b;
    if (bright < 128) return CSI "40m";   /* Black */
    if (c.r > 200 && c.g < 100 && c.b < 100) return CSI "41m";  /* Red */
    if (c.r < 100 && c.g > 200 && c.b < 100) return CSI "42m";  /* Green */
    if (c.r > 200 && c.g > 200 && c.b < 100) return CSI "43m";  /* Yellow */
    if (c.r < 100 && c.g < 100 && c.b > 200) return CSI "44m";  /* Blue */
    if (c.r > 200 && c.g < 100 && c.b > 200) return CSI "45m";  /* Magenta */
    if (c.r < 100 && c.g > 200 && c.b > 200) return CSI "46m";  /* Cyan */
    return CSI "47m";  /* White */
}

void display_init(void) {
    printf(HIDE_CURSOR);
    printf(CLEAR_SCREEN);
    fflush(stdout);
}

void display_cleanup(void) {
    printf(SHOW_CURSOR);
    printf(CURSOR_HOME);
    printf(CLEAR_SCREEN);
    fflush(stdout);
}

void display_render(const RGB *frame, int w, int h,
                    const Detection *det, float fps,
                    int pan, int tilt) {
    printf(CURSOR_HOME);

    /* Render frame — scale to fit terminal width (~80 chars = 40 pixel columns) */
    int display_w = w;
    int display_h = h;

    /* If too wide, halve the width (each pixel = 2 chars) */
    if (display_w * 2 > 78) {
        display_w = 39;
        display_h = h * 39 / w;
    }

    /* Precompute bounding box in display coords */
    int bx = 0, by = 0, bw = 0, bh = 0;
    if (det && det->detected) {
        bx = det->x * display_w / w;
        by = det->y * display_h / h;
        bw = det->w * display_w / w;
        bh = det->h * display_h / h;
    }

    for (int y = 0; y < display_h; y++) {
        int src_y = y * h / display_h;
        for (int x = 0; x < display_w; x++) {
            int src_x = x * w / display_w;
            RGB c = frame[src_y * w + src_x];

            /* Check if this pixel is on bounding box border */
            int on_border = (det && det->detected &&
                (x == bx || x == bx + bw - 1 ||
                 y == by || y == by + bh - 1) &&
                x >= bx && x < bx + bw && y >= by && y < by + bh);
            
            if (on_border) {
                /* Bounding box border */
                printf(CSI "42m  " RESET);  /* Green border */
            } else {
                printf("%s  " RESET, rgb_to_ansi_bg(c));
            }
        }
        printf("\n");
    }

    /* Status bar */
    printf("─── tinydrone sim ─── ");
    if (det && det->detected) {
        printf("TARGET at (%d,%d) area=%d conf=%.2f  ",
               det->cx, det->cy, det->area, det->confidence);
    } else {
        printf("NO TARGET                         ");
    }
    if (fps > 0) printf("FPS: %.1f  ", fps);
    printf("Servo: P%d T%d", pan, tilt);
    printf("\n");

    fflush(stdout);
}

void display_render_mask(const uint8_t *mask, int w, int h,
                         const Detection *det) {
    (void)det;
    printf(CURSOR_HOME);

    int display_w = w;
    int display_h = h;
    if (display_w * 2 > 78) {
        display_w = 39;
        display_h = h * 39 / w;
    }

    for (int y = 0; y < display_h; y++) {
        int src_y = y * h / display_h;
        for (int x = 0; x < display_w; x++) {
            int src_x = x * w / display_w;
            if (mask[src_y * w + src_x]) {
                printf(CSI "47m  " RESET);  /* White = detected */
            } else {
                printf(CSI "40m  " RESET);  /* Black = background */
            }
        }
        printf("\n");
    }

    printf("─── MASK VIEW ───\n");
    fflush(stdout);
}
