# TINYDRONE — simülasyon CNN köprüsü
# libtinydrone.so: inference_int8.c + sliding_window.c + bridge.c
# (ESP32 firmware'deki kodun aynısı — host'ta paylaşımlı kütüphane)

CC      ?= cc
CFLAGS  ?= -std=c11 -O2 -Wall -fPIC -I../firmware/main -I../training/output
LDFLAGS ?= -shared -lm

SRCS = ../firmware/main/inference_int8.c ../firmware/main/sliding_window.c bridge.c
TARGET = libtinydrone.so

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $@ $(LDFLAGS)
	@echo "==> $@ hazır"

test: $(TARGET)
	python3 test_bridge.py

clean:
	rm -f $(TARGET) __pycache__ -rf

.PHONY: all test clean
