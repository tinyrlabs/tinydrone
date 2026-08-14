/**
 * uart_out.c — hedef offset çıkışı (FC entegrasyonu)
 */

#include "uart_out.h"
#include <stdio.h>

#include "driver/uart.h"
#include "driver/gpio.h"

#define UART_PORT  UART_NUM_2
#define UART_TX_GPIO GPIO_NUM_4
#define UART_RX_GPIO GPIO_NUM_5
#define UART_BAUD  115200
#define UART_BUF   256

void uart_out_init(void) {
    uart_config_t cfg = {
        .baud_rate = UART_BAUD,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
    };
    uart_param_config(UART_PORT, &cfg);
    uart_set_pin(UART_PORT, UART_TX_GPIO, UART_RX_GPIO, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
    uart_driver_install(UART_PORT, UART_BUF, UART_BUF, 0, NULL, 0);
}

void uart_out_send_offset(double offset_x, double offset_y) {
    /* [-1,1] → [-100,+100] int */
    int ix = (int)(offset_x * 100.0 + (offset_x >= 0 ? 0.5 : -0.5));
    int iy = (int)(offset_y * 100.0 + (offset_y >= 0 ? 0.5 : -0.5));
    if (ix > 100) ix = 100;
    if (ix < -100) ix = -100;
    if (iy > 100) iy = 100;
    if (iy < -100) iy = -100;

    char buf[16];
    int n = snprintf(buf, sizeof(buf), "T%+03dY%+03d\n", ix, iy);
    if (n > 0) uart_write_bytes(UART_PORT, buf, n);
}

void uart_out_send_lost(void) {
    const char *msg = "T+000Y+000\n";  /* merkez = hedef yok */
    uart_write_bytes(UART_PORT, msg, 10);
}
