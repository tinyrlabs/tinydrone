/**
 * uart_out.h — hedef offset çıkışı (FC entegrasyonu)
 *
 * UART2: GPIO4 (TXD), GPIO5 (RXD), 115200 8N1
 * Format: T<offset_x>Y<offset_y>\n  (offset -100..+100)
 * Örnek:  T+035Y-012\n  → hedef sağda 0.35, yukarıda 0.12
 */

#ifndef TINYDRONE_UART_OUT_H
#define TINYDRONE_UART_OUT_H

#include <stdint.h>

/** UART2 başlat (115200 8N1, GPIO4 TX / GPIO5 RX). */
void uart_out_init(void);

/**
 * Hedef offset gönder.
 * @param offset_x  [-1.0, 1.0] — + = sağda
 * @param offset_y  [-1.0, 1.0] — + = yukarıda
 */
void uart_out_send_offset(double offset_x, double offset_y);

/** Hedef kayıp sinyali (opsiyonel — FC'ye "hedef yok" bildirimi). */
void uart_out_send_lost(void);

#endif /* TINYDRONE_UART_OUT_H */
