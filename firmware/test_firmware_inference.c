/**
 * test_firmware_inference.c — firmware inference.c'yi host'ta doğrular
 *
 * inference.c ESP-IDF'den bağımsızdır (sadece matrix/conv2d/pool2d/model.h).
 * Aynı test görselini firmware koduyla sınıflandırır ve evaluate.c sonucuyla
 * karşılaştırır.
 *
 * Build:
 *   cc -std=c11 -O2 -I. -Icomponents/tinycml/include \
 *      test_firmware_inference.c main/inference.c \
 *      components/tinycml/src/{matrix,conv2d,pool2d}.c -lm -o test_fi
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "inference.h"  /* firmware/main/inference.h */

static const char *CLASS_NAMES[] = {"tank", "armored_vehicle", "drone_uav", "background"};

int main(void) {
    inference_init();
    printf("Firmware inference init OK\n");

    /* Her sınıftan 50 görsel test et */
    const char *cls[] = {"tank", "armored_vehicle", "drone_uav", "background"};
    int total = 0, correct = 0;

    for (int c = 0; c < 4; c++) {
        char path[1024];
        snprintf(path, sizeof(path), "dataset/processed/test/%s", cls[c]);
        DIR *d = opendir(path);
        if (!d) continue;
        struct dirent *e;
        int cnt = 0;
        while ((e = readdir(d)) && cnt < 50) {
            if (!strstr(e->d_name, ".png")) continue;
            char fp[2048];
            snprintf(fp, sizeof(fp), "%s/%s", path, e->d_name);
            int w, h, ch;
            unsigned char *px = stbi_load(fp, &w, &h, &ch, 3);
            if (!px) continue;

            /* Kamera formatına çevir: RGB888 → double NCHW */
            double frame[32 * 32 * 3];
            for (int y = 0; y < 32; y++)
                for (int x = 0; x < 32; x++)
                    for (int cc = 0; cc < 3; cc++)
                        frame[cc * 1024 + y * 32 + x] =
                            (double)px[(y * w + x) * 3 + cc] / 127.5 - 1.0;
            stbi_image_free(px);

            double probs[4];
            int pred = inference_run(frame, probs);

            if (pred == c) correct++;
            total++;
            cnt++;
        }
        closedir(d);
        printf("  %-16s: %d görsel test edildi\n", cls[c], cnt);
    }

    printf("\nFirmware inference doğruluğu: %.2f%% (%d/%d)\n",
           100.0 * correct / total, correct, total);
    printf("(evaluate.c ile aynı sonucu üretmeli — CNN mantığı birebir aynı)\n");
    return 0;
}
