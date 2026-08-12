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
#include "inference.h"
#include "tracker.h"

static const char *TAG = "tinydrone";

/* Askeri hedef sınıfları (background hariç) */
#define CLASS_TANK          0
#define CLASS_ARMORED       1
#define CLASS_DRONE         2
#define CLASS_BACKGROUND    3

/* Takip edilecek sınıflar */
static const int TARGET_CLASSES[] = {CLASS_TANK, CLASS_ARMORED, CLASS_DRONE};
#define N_TARGET_CLASSES 3

/* Hedef güven eşiği — %60 üstü askeri hedef say */
#define CONF_THRESHOLD 0.60

void app_main(void) {
    ESP_LOGI(TAG, "tinydrone başlatılıyor...");

    if (camera_init() != 0) {
        ESP_LOGE(TAG, "Kamera init başarısız");
        return;
    }
    ESP_LOGI(TAG, "Kamera OK (QQVGA RGB565)");

    inference_init();
    ESP_LOGI(TAG, "tinycml CNN modeli yüklendi");

    tracker_init();
    ESP_LOGI(TAG, "Servolar OK");

    double frame[32 * 32 * 3];
    double probs[4];
    uint32_t frame_count = 0;

    while (1) {
        int64_t t0 = esp_timer_get_time();

        /* 1. Kare al */
        if (camera_capture_32x32(frame) != 0) {
            ESP_LOGW(TAG, "Kare alınamadı");
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        /* 2. Sınıflandır */
        int pred = inference_run(frame, probs);
        frame_count++;

        /* 3. Takip kararı */
        if (pred >= 0 && pred < N_TARGET_CLASSES && probs[pred] >= CONF_THRESHOLD) {
            /* Basit takip: hedef merkezde varsay, servo sabit.
             * Gerçek bounding box tespiti için sliding window gerekir (Faz 4.5) */
            tracker_update(0.0, 0.0);
            ESP_LOGI(TAG, "HEDEF: sınıf=%d güven=%.2f", pred, probs[pred]);
        } else if (frame_count % 100 == 0) {
            /* Her 100 karede bir durum logu */
            ESP_LOGI(TAG, "frame=%lu pred=%d probs=[%.2f %.2f %.2f %.2f]",
                     frame_count, pred, probs[0], probs[1], probs[2], probs[3]);
        }

        int64_t t1 = esp_timer_get_time();
        int fps = 1000000 / (t1 - t0 > 0 ? t1 - t0 : 1);
        if (frame_count % 200 == 0) {
            ESP_LOGI(TAG, "FPS: %d", fps);
        }

        vTaskDelay(pdMS_TO_TICKS(10));  /* ~100 FPS hedef, inference ağır basar */
    }
}
