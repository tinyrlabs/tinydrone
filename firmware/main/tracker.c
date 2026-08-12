/**
 * tracker.c — servo tabanlı hedef takibi (LEDC PWM)
 *
 * SG90 mikro servo: 50Hz, 500us (0°) - 2500us (180°)
 *   GPIO14 = pan, GPIO15 = tilt
 */

#include "tracker.h"
#include <math.h>

#include "driver/ledc.h"

#define SERVO_FREQ     50        /* Hz */
#define SERVO_MIN_US   500
#define SERVO_MAX_US   2500
#define SERVO_RANGE_DEG 180

static double pan_deg = 90.0;
static double tilt_deg = 90.0;

static void servo_set_us(ledc_channel_t ch, double us) {
    /* LEDC timer 13-bit çözünürlük: max 8191 */
    uint32_t duty = (uint32_t)(us * 8191.0 / (1000000.0 / SERVO_FREQ));
    ledc_set_duty(LEDC_LOW_SPEED_MODE, ch, duty);
    ledc_update_duty(LEDC_LOW_SPEED_MODE, ch);
}

void tracker_init(void) {
    ledc_timer_config_t timer = {
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .timer_num = LEDC_TIMER_0,
        .duty_resolution = LEDC_TIMER_13_BIT,
        .freq_hz = SERVO_FREQ,
        .clk_cfg = LEDC_AUTO_CLK,
    };
    ledc_timer_config(&timer);

    ledc_channel_config_t pan = {
        .gpio_num = 14,
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .channel = LEDC_CHANNEL_0,
        .timer_sel = LEDC_TIMER_0,
        .duty = 0,
        .hpoint = 0,
    };
    ledc_channel_config(&pan);

    ledc_channel_config_t tilt = {
        .gpio_num = 15,
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .channel = LEDC_CHANNEL_1,
        .timer_sel = LEDC_TIMER_0,
        .duty = 0,
        .hpoint = 0,
    };
    ledc_channel_config(&tilt);

    tracker_update(0.0, 0.0);  /* merkeze al */
}

void tracker_update(double offset_x, double offset_y) {
    /* PID yerine basit P kontrolcüsü — kazanç 30°/birim */
    pan_deg -= offset_x * 30.0;
    tilt_deg += offset_y * 30.0;

    if (pan_deg < 0.0) pan_deg = 0.0;
    if (pan_deg > 180.0) pan_deg = 180.0;
    if (tilt_deg < 0.0) tilt_deg = 0.0;
    if (tilt_deg > 180.0) tilt_deg = 180.0;

    double pan_us = SERVO_MIN_US + (pan_deg / SERVO_RANGE_DEG) * (SERVO_MAX_US - SERVO_MIN_US);
    double tilt_us = SERVO_MIN_US + (tilt_deg / SERVO_RANGE_DEG) * (SERVO_MAX_US - SERVO_MIN_US);

    servo_set_us(LEDC_CHANNEL_0, pan_us);
    servo_set_us(LEDC_CHANNEL_1, tilt_us);
}

void tracker_get_angles(double *pan, double *tilt) {
    if (pan) *pan = pan_deg;
    if (tilt) *tilt = tilt_deg;
}
