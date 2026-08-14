/**
 * compare_i8.c — float vs int8 doğruluk karşılaştırması
 *
 * Aynı test görsellerini hem float hem int8 modelle sınıflandırır,
 * doğruluk ve uyum (agreement) oranını raporlar.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "float_inference.h"      /* float */
#include "inference_int8.h" /* int8 */

int main(void) {
    inference_init();
    inference_int8_init();
    printf("Model'ler yüklendi (float + int8)\n\n");

    const char *cls[] = {"tank", "armored_vehicle", "drone_uav", "background"};
    int f_total = 0, f_correct = 0;
    int i_total = 0, i_correct = 0;
    int agree = 0;

    for (int c = 0; c < 4; c++) {
        char path[1024];
        snprintf(path, sizeof(path), "dataset/processed/test/%s", cls[c]);
        DIR *d = opendir(path);
        if (!d) continue;
        struct dirent *e;
        int cnt = 0;
        while ((e = readdir(d)) && cnt < 200) {
            if (!strstr(e->d_name, ".png")) continue;
            char fp[2048];
            snprintf(fp, sizeof(fp), "%s/%s", path, e->d_name);
            int w, h, ch;
            unsigned char *px = stbi_load(fp, &w, &h, &ch, 3);
            if (!px) continue;

            double frame[32 * 32 * 3];
            for (int y = 0; y < 32; y++)
                for (int x = 0; x < 32; x++)
                    for (int cc = 0; cc < 3; cc++)
                        frame[cc * 1024 + y * 32 + x] =
                            (double)px[(y * w + x) * 3 + cc] / 127.5 - 1.0;
            stbi_image_free(px);

            double fp_probs[4], i8_probs[4];
            int fp_pred = inference_run(frame, fp_probs);
            int i8_pred = inference_int8_run(frame, i8_probs);

            f_total++; i_total++;
            if (fp_pred == c) f_correct++;
            if (i8_pred == c) i_correct++;
            if (fp_pred == i8_pred) agree++;
            cnt++;
        }
        closedir(d);
        printf("  %-16s tamam\n", cls[c]);
    }

    printf("\n================ SONUÇ ================\n");
    printf("Float model:  %6.2f%% (%d/%d)\n", 100.0 * f_correct / f_total, f_correct, f_total);
    printf("int8 model:   %6.2f%% (%d/%d)\n", 100.0 * i_correct / i_total, i_correct, i_total);
    printf("Uyum (agreement): %6.2f%% (%d/%d)\n", 100.0 * agree / f_total, agree, f_total);
    printf("Doğruluk farkı: %+.2f puan\n", 100.0 * (i_correct - f_correct) / f_total);

    return 0;
}
