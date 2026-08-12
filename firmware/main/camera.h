/**
 * camera.h — OV2640 kamera wrapper (ESP32-CAM)
 *
 * ESP32-CAM (AI-Thinker) pin konfigürasyonu:
 *   PWDN=GPIO32, RESET=-1, XCLK=GPIO0, SIOD=GPIO26, SIOC=GPIO27
 *   Y9=GPIO35, Y8=GPIO34, Y7=GPIO39, Y6=GPIO36, Y5=GPIO21
 *   Y4=GPIO19, Y3=GPIO18, Y2=GPIO5, VSYNC=GPIO25, HREF=GPIO23, PCLK=GPIO22
 */

#ifndef TINYDRONE_CAMERA_H
#define TINYDRONE_CAMERA_H

#include <stdint.h>

/** Initialize OV2640 at QQVGA (160x120) grayscale-friendly mode */
int camera_init(void);

/**
 * Capture a frame and downsample to 32x32 RGB (NCHW order, [-1,1]).
 *
 * @param out  32x32x3 buffer (3072 doubles), normalized to [-1, 1]
 * @return 0 on success, -1 on failure
 */
int camera_capture_32x32(double *out);

/**
 * Capture raw RGB888 frame (160x120, row-major, 3 bytes/pixel).
 * Sliding window tespiti için.
 *
 * @param out  buffer (160*120*3 bytes)
 * @return 0 on success, -1 on failure
 */
int camera_capture_frame(uint8_t *out);

#endif /* TINYDRONE_CAMERA_H */
