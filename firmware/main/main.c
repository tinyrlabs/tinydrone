/**
 * main.c — tinydrone ESP32-CAM firmware
 *
 * Döngü:
 *   1. Kamera karesi al (QQVGA 160x120 RGB565)
 *   2. 32x32x3'e küçült
 *   3. tinycml CNN ile sınıflandır (4 sınıf)
 *   4. Askeri hedef tespit edilirse servo ile takip et
 *
 * Sınıf indexleri: 0=tank, 1=armored_vehicle, 2=drone_uav, 3=background
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"

#include "camera.h"
#include "inference_int8.h"  /* int8 — ESP32 için 4x hızlı */
#include "sliding_window.h"
#include "tracker.h"
#include "uart_out.h"

static const char *TAG = "tinydrone";

/* Askeri hedef sınıfları (background hariç) */
#define CLASS_TANK          0
#define CLASS_ARMORED       1
#define CLASS_DRONE         2
#define CLASS_BACKGROUND    3

/* Takip eşiği — %60 üstü askeri hedef say */
#define CONF_THRESHOLD 0.60

/* int8 inference callback (sliding window için) */
static int sw_infer_cb(const double *patch, double *probs) {
    return inference_int8_run(patch, probs);
}

void app_main(void) {
    ESP_LOGI(TAG, "tinydrone başlatılıyor...");

    if (camera_init() != 0) {
        ESP_LOGE(TAG, "Kamera init başarısız");
        return;
    }
    ESP_LOGI(TAG, "Kamera OK (QQVGA RGB565)");

    inference_int8_init();
    ESP_LOGI(TAG, "tinycml CNN modeli yüklendi (int8)");

    tracker_init();
    ESP_LOGI(TAG, "Servolar OK");

    uart_out_init();
    ESP_LOGI(TAG, "UART çıkışı OK (GPIO4/5, 115200)");

    uint8_t frame[SW_FRAME_W * SW_FRAME_H * 3];
    uint32_t frame_count = 0;
    SWDetection det;
    SWTracker trk;
    sw_tracker_init(&trk);
    int uart_sent_lost = 0;

    while (1) {
        int64_t t0 = esp_timer_get_time();

        /* 1. Kare al (tam çözünürlük) */
        if (camera_capture_frame(frame) != 0) {
            ESP_LOGW(TAG, "Kare alınamadı");
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        /* 2. Tespit + takip (kilitliyken hızlı, kayıpta tam tarama) */
        sw_detect_track(frame, sw_infer_cb, &trk, &det);
        frame_count++;

        /* 3. Takip kararı + UART çıkışı */
        if (det.detected && det.confidence >= CONF_THRESHOLD) {
            tracker_update(det.offset_x, det.offset_y);
            uart_out_send_offset(det.offset_x, det.offset_y);
            uart_sent_lost = 0;
            ESP_LOGI(TAG, "HEDEF: sınıf=%d güven=%.2f bbox=(%d,%d) off=(%.2f,%.2f) kilit=%d",
                     det.cls, det.confidence, det.x, det.y, det.offset_x, det.offset_y,
                     trk.locked);
        } else {
            if (!uart_sent_lost) {
                uart_out_send_lost();
                uart_sent_lost = 1;
            }
            if (frame_count % 20 == 0) {
                ESP_LOGI(TAG, "tarama: hedef yok (frame=%lu kilit=%d)", frame_count, trk.locked);
            }
        }

        int64_t t1 = esp_timer_get_time();
        int fps = 1000000 / (t1 - t0 > 0 ? t1 - t0 : 1);
        if (frame_count % 50 == 0) {
            ESP_LOGI(TAG, "FPS: %d", fps);
        }

        vTaskDelay(pdMS_TO_TICKS(10));
    }
}
