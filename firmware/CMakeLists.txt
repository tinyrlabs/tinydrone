# tinydrone — ESP32-CAM Firmware
#
# Build (ESP-IDF v5.x):
#   export IDF_PATH=~/esp-idf
#   . $IDF_PATH/export.sh
#   idf.py set-target esp32
#   idf.py build
#   idf.py -p /dev/ttyUSB0 flash monitor
#
# Donanım: AI-Thinker ESP32-CAM (OV2640)
#   GPIO4  = Kamera PWDN
#   GPIO0  = Flash LED
#   GPIO14 = Servo pan (PWM)
#   GPIO15 = Servo tilt (PWM)

cmake_minimum_required(VERSION 3.16)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(tinydrone)
