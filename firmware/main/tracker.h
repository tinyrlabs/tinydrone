/**
 * tracker.h — servo tabanlı hedef takibi
 *
 * İki servo: pan (yatay) + tilt (dikey). LEDC PWM ile sürülür.
 *   GPIO14 = pan, GPIO15 = tilt
 */

#ifndef TINYDRONE_TRACKER_H
#define TINYDRONE_TRACKER_H

/** Init servolar (50Hz, 500-2500us) */
void tracker_init(void);

/**
 * Hedefi merkeze al — bounding box offset'ine göre servo açılarını güncelle.
 *
 * @param offset_x  [-1, 1] — hedefin görüntü merkezine göre yatay sapması
 * @param offset_y  [-1, 1] — dikey sapma
 */
void tracker_update(double offset_x, double offset_y);

/** Servo açılarını döndür (derece) */
void tracker_get_angles(double *pan, double *tilt);

#endif /* TINYDRONE_TRACKER_H */
