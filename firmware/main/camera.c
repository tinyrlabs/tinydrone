/**
 * camera.c — OV2640 kamera wrapper (ESP32-CAM)
 *
 * AI-Thinker ESP32-CAM pinout:
 *   PWDN=32, RESET=-1, XCLK=0, SIOD=26, SIOC=27
 *   Y9=35, Y8=34, Y7=39, Y6=36, Y5=21
 *   Y4=19, Y3=18, Y2=5, VSYNC=25, HREF=23, PCLK=22
 */

#include "camera.h"
#include <string.h>

#include "esp_camera.h"

/* ESP32-CAM pin tanımları */
#define CAM_PIN_PWDN    32
#define CAM_PIN_RESET   -1
#define CAM_PIN_XCLK    0
#define CAM_PIN_SIOD    26
#define CAM_PIN_SIOC    27
#define CAM_PIN_D7      35
#define CAM_PIN_D6      34
#define CAM_PIN_D5      39
#define CAM_PIN_D4      36
#define CAM_PIN_D3      21
#define CAM_PIN_D2      19
#define CAM_PIN_D1      18
#define CAM_PIN_D0      5
#define CAM_PIN_VSYNC   25
#define CAM_PIN_HREF    23
#define CAM_PIN_PCLK    22

int camera_init(void) {
    camera_config_t config = {
        .pin_pwdn  = CAM_PIN_PWDN,
        .pin_reset = CAM_PIN_RESET,
        .pin_xclk = CAM_PIN_XCLK,
        .pin_sccb_sda = CAM_PIN_SIOD,
        .pin_sccb_scl = CAM_PIN_SIOC,
        .pin_d7 = CAM_PIN_D7,
        .pin_d6 = CAM_PIN_D6,
        .pin_d5 = CAM_PIN_D5,
        .pin_d4 = CAM_PIN_D4,
        .pin_d3 = CAM_PIN_D3,
        .pin_d2 = CAM_PIN_D2,
        .pin_d1 = CAM_PIN_D1,
        .pin_d0 = CAM_PIN_D0,
        .pin_vsync = CAM_PIN_VSYNC,
        .pin_href = CAM_PIN_HREF,
        .pin_pclk = CAM_PIN_PCLK,

        /* QQVGA 160x120 — CNN için düşük çözünürlük, hızlı */
        .xclk_freq_hz = 20000000,
        .ledc_timer = LEDC_TIMER_0,
        .ledc_channel = LEDC_CHANNEL_0,
        .pixel_format = PIXFORMAT_RGB565,  /* renk bilgisi için */
        .frame_size = FRAMESIZE_QQVGA,
        .jpeg_quality = 12,
        .fb_count = 1,
        .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
        .fb_location = CAMERA_FB_IN_PSRAM,
    };

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) return -1;

    sensor_t *s = esp_camera_sensor_get();
    if (s) {
        s->set_hmirror(s, 0);
        s->set_vflip(s, 0);
        s->set_brightness(s, 0);
        s->set_saturation(s, 0);
    }
    return 0;
}

int camera_capture_32x32(double *out) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) return -1;

    if (fb->format != PIXFORMAT_RGB565 || fb->width != 160 || fb->height != 120) {
        esp_camera_fb_return(fb);
        return -1;
    }

    /* RGB565 → 32x32 downsampling → NCHW [-1,1] */
    /* OV2640 RGB565: little-endian, 16-bit: RRRRRGGGGGGBBBBB */
    uint16_t *pixels = (uint16_t *)fb->buf;
    const int src_w = 160, src_h = 120;

    for (int y = 0; y < 32; y++) {
        for (int x = 0; x < 32; x++) {
            int sx = x * src_w / 32;
            int sy = y * src_h / 32;
            uint16_t p = pixels[sy * src_w + sx];
            uint8_t r = (p >> 11) & 0x1F;
            uint8_t g = (p >> 5) & 0x3F;
            uint8_t b = p & 0x1F;
            /* 5/6-bit → 8-bit, sonra [-1, 1] */
            out[0 * 1024 + y * 32 + x] = (double)(r << 3) / 127.5 - 1.0;
            out[1 * 1024 + y * 32 + x] = (double)(g << 2) / 127.5 - 1.0;
            out[2 * 1024 + y * 32 + x] = (double)(b << 3) / 127.5 - 1.0;
        }
    }

    esp_camera_fb_return(fb);
    return 0;
}

int camera_capture_frame(uint8_t *out) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) return -1;

    if (fb->format != PIXFORMAT_RGB565 || fb->width != 160 || fb->height != 120) {
        esp_camera_fb_return(fb);
        return -1;
    }

    /* RGB565 → RGB888 */
    uint16_t *pixels = (uint16_t *)fb->buf;
    for (int i = 0; i < 160 * 120; i++) {
        uint16_t p = pixels[i];
        out[i * 3 + 0] = (p >> 11) & 0x1F;
        out[i * 3 + 1] = (p >> 5) & 0x3F;
        out[i * 3 + 2] = p & 0x1F;
        out[i * 3 + 0] <<= 3;  /* 5-bit → 8-bit */
        out[i * 3 + 1] <<= 2;  /* 6-bit → 8-bit */
        out[i * 3 + 2] <<= 3;
    }

    esp_camera_fb_return(fb);
    return 0;
}
