/**
 * main.c - tinydrone desktop simulation
 *
 * Generates synthetic test frames with a moving colored target
 * and runs the detection + tracking pipeline.
 *
 * Build: make sim && ./build/sim_tinydrone
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>

#include "detector.h"
#include "display.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================
 * Synthetic frame generator (for testing)
 * ============================================ */

static void draw_target(RGB *frame, int w, int h,
                        int cx, int cy, int radius,
                        RGB color) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int dx = x - cx;
            int dy = y - cy;
            if (dx * dx + dy * dy <= radius * radius) {
                frame[y * w + x] = color;
            } else {
                /* Background: dark green grass-like */
                RGB bg = {20, 60 + (y % 20), 20 + (x % 10)};
                frame[y * w + x] = bg;
            }
        }
    }
}

/* ============================================
 * Main loop
 * ============================================ */

static volatile int running = 1;

static void sigint_handler(int sig) {
    (void)sig;
    running = 0;
}

int main(void) {
    signal(SIGINT, sigint_handler);

    printf("tinydrone — Desktop Simulation\n");
    printf("===============================\n");
    printf("Press Ctrl+C to exit\n\n");

    /* Allocate buffers */
    int sim_w = 320, sim_h = 240;
    int ds_w = TD_DOWNSAMPLE_W, ds_h = TD_DOWNSAMPLE_H;

    RGB *frame = malloc((size_t)sim_w * sim_h * sizeof(RGB));
    RGB *ds_frame = malloc((size_t)ds_w * ds_h * sizeof(RGB));
    uint8_t *mask = malloc((size_t)ds_w * ds_h);
    if (!frame || !ds_frame || !mask) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    /* Target color: bright red ball */
    ColorRange red_ball = {
        .h_min = 0,   .h_max = 10,    /* Red wraps around 0 */
        .s_min = 100, .s_max = 255,   /* High saturation */
        .v_min = 100, .v_max = 255    /* Bright */
    };

    /* Alternative: also catch red near 170-179 */
    ColorRange red_ball2 = {
        .h_min = 170, .h_max = 179,
        .s_min = 100, .s_max = 255,
        .v_min = 100, .v_max = 255
    };

    Tracker tracker;
    tracker_init(&tracker);

    display_init();

    /* Moving target parameters */
    float angle = 0.0f;
    int target_cx, target_cy;
    int frame_count = 0;
    clock_t start_time = clock();

    while (running) {
        /* Move target in a circle + lissajous pattern */
        target_cx = sim_w / 2 + (int)(sim_w / 3 * cos(angle));
        target_cy = sim_h / 2 + (int)(sim_h / 4 * sin(angle * 2.3));
        angle += 0.05f;
        if (angle > 2 * M_PI) angle -= 2 * M_PI;

        /* Draw synthetic frame */
        RGB target_color = {220, 40, 30};  /* Red ball */
        draw_target(frame, sim_w, sim_h, target_cx, target_cy, 25, target_color);

        /* Downsample */
        downsample_rgb(frame, sim_w, sim_h, ds_frame, ds_w, ds_h);

        /* Detect */
        Detection result1, result2;
        detector_process(ds_frame, ds_w, ds_h, &red_ball, &result1);
        detector_process(ds_frame, ds_w, ds_h, &red_ball2, &result2);

        /* Use whichever has larger area */
        Detection *best = &result1;
        if (result2.detected && (!result1.detected || result2.area > result1.area)) {
            best = &result2;
        }

        /* Track */
        int dpan, dtilt;
        tracker_update(&tracker, best, &dpan, &dtilt);

        /* FPS */
        frame_count++;
        clock_t now = clock();
        float elapsed = (float)(now - start_time) / CLOCKS_PER_SEC;
        float fps = elapsed > 0 ? frame_count / elapsed : 0;

        /* Display */
        display_render(ds_frame, ds_w, ds_h, best, fps,
                       tracker.pan_angle, tracker.tilt_angle);

        /* Simulate ~30fps */
        struct timespec ts = {0, 33000000};  /* 33ms */
        nanosleep(&ts, NULL);
    }

    display_cleanup();

    /* Report */
    printf("\n");
    printf("Simulation complete.\n");
    printf("Frames processed: %d\n", frame_count);
    printf("Final servo: Pan=%d Tilt=%d\n",
           tracker.pan_angle, tracker.tilt_angle);

    free(frame);
    free(ds_frame);
    free(mask);

    return 0;
}
