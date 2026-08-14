# tinydrone — Build System
#
# Targets:
#   make sim        — Desktop simulation (color-based detector)
#   make demo       — CNN terminal demo (model tahminleri + ASCII)
#   make evaluate   — Test seti doğruluğu
#   make confusion  — Confusion matrix analizi
#   make test       — Build and run unit tests
#   make clean

CC        = cc
CFLAGS    = -std=c11 -Wall -Wextra -pedantic -O2 -g -D_POSIX_C_SOURCE=200809L
LDFLAGS   = -lm

BUILD_DIR  = build
SIM_DIR    = simulation
TRAIN_DIR  = training
TINYCML    = /home/ubuntu/projects/tinycml
TINYCML_INC = $(TINYCML)/include
TINYCML_LIB = $(TINYCML)/build/lib/libtinycml.a
MODEL_DIR  = $(TRAIN_DIR)/output

SIM_SRCS   = $(SIM_DIR)/main.c $(SIM_DIR)/detector.c $(SIM_DIR)/display.c
SIM_OBJS   = $(patsubst $(SIM_DIR)/%.c,$(BUILD_DIR)/%.o,$(SIM_SRCS))
SIM_TARGET = $(BUILD_DIR)/sim_tinydrone

DEMO_SRC     = $(TRAIN_DIR)/demo.c
EVAL_SRC     = $(TRAIN_DIR)/evaluate.c
CONF_SRC     = $(TRAIN_DIR)/confusion.c
DEMO_TARGET  = $(BUILD_DIR)/demo
EVAL_TARGET  = $(BUILD_DIR)/evaluate
CONF_TARGET  = $(BUILD_DIR)/confusion

TEST_DIR   = tests
TEST_SRCS  = $(wildcard $(TEST_DIR)/*.c)
TEST_BINS  = $(patsubst $(TEST_DIR)/%.c,$(BUILD_DIR)/test_%,$(TEST_SRCS))

.PHONY: all sim demo evaluate confusion test clean

all: sim demo evaluate confusion

sim: $(SIM_TARGET)

$(SIM_TARGET): $(SIM_OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "  BUILD  $@"
	@echo "  Run: ./$(SIM_TARGET)"

$(BUILD_DIR)/%.o: $(SIM_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(SIM_DIR) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# CNN araçları (model header output/ içinde olmalı)
demo: $(DEMO_TARGET)

$(DEMO_TARGET): $(DEMO_SRC) $(TINYCML_LIB) $(MODEL_DIR)/tinydrone_model.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(TINYCML_INC) -I$(TRAIN_DIR) -I$(MODEL_DIR) \
		$(DEMO_SRC) $(TINYCML_LIB) -o $@ $(LDFLAGS)
	@echo "  BUILD  $@"
	@echo "  Run: ./$(DEMO_TARGET) training/dataset/processed"

evaluate: $(EVAL_TARGET)

$(EVAL_TARGET): $(EVAL_SRC) $(TINYCML_LIB) $(MODEL_DIR)/tinydrone_model.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(TINYCML_INC) -I$(TRAIN_DIR) -I$(MODEL_DIR) \
		$(EVAL_SRC) $(TINYCML_LIB) -o $@ $(LDFLAGS)
	@echo "  BUILD  $@"
	@echo "  Run: ./$(EVAL_TARGET) training/dataset/processed"

confusion: $(CONF_TARGET)

$(CONF_TARGET): $(CONF_SRC) $(TINYCML_LIB) $(MODEL_DIR)/tinydrone_model.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(TINYCML_INC) -I$(TRAIN_DIR) -I$(MODEL_DIR) \
		$(CONF_SRC) $(TINYCML_LIB) -o $@ $(LDFLAGS)
	@echo "  BUILD  $@"
	@echo "  Run: ./$(CONF_TARGET) training/dataset/processed"

test: $(TEST_BINS)
	@passed=0; total=0; \
	for t in $(TEST_BINS); do \
		./$$t && passed=$$((passed+1)); \
		total=$$((total+1)); \
	done; \
	echo "$$passed/$$total tests passed"

$(BUILD_DIR)/test_%: $(TEST_DIR)/%.c $(SIM_DIR)/detector.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(SIM_DIR) $< $(SIM_DIR)/detector.c -o $@ $(LDFLAGS)

# Firmware host testleri (takip modu + int8 doğrulama)
FW_DIR      = firmware
FW_INC      = -I$(FW_DIR) -I$(FW_DIR)/main -I$(FW_DIR)/components/tinycml/include
FW_CML_SRCS = $(FW_DIR)/components/tinycml/src/matrix.c \
              $(FW_DIR)/components/tinycml/src/conv2d.c \
              $(FW_DIR)/components/tinycml/src/pool2d.c \
              $(FW_DIR)/components/tinycml/src/cml_error.c

test-host: $(BUILD_DIR)/test_track $(BUILD_DIR)/compare_i8
	@ln -sfn ../$(TRAIN_DIR)/dataset $(FW_DIR)/dataset
	@cd $(FW_DIR) && ../$(BUILD_DIR)/test_track
	@cd $(TRAIN_DIR) && ../$(BUILD_DIR)/compare_i8
	@rm -f $(FW_DIR)/dataset
	@echo "  HOST TESTS OK"

$(BUILD_DIR)/test_track: $(FW_DIR)/test_track.c $(FW_DIR)/main/sliding_window.c \
                         $(FW_DIR)/main/inference_int8.c $(FW_CML_SRCS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(FW_INC) -I$(TRAIN_DIR) $^ -o $@ $(LDFLAGS)

$(BUILD_DIR)/compare_i8: $(TRAIN_DIR)/compare_i8.c $(TRAIN_DIR)/float_inference.c \
                         $(FW_DIR)/main/inference_int8.c $(TINYCML_LIB) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(TINYCML_INC) -I$(TRAIN_DIR) -I$(MODEL_DIR) -I$(FW_DIR)/main \
		$^ -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)
