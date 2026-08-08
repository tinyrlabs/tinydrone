# tinydrone — Build System
#
# Targets:
#   make sim       — Build desktop simulation
#   make test      — Build and run unit tests
#   make clean

CC       = cc
CFLAGS   = -std=c11 -Wall -Wextra -pedantic -O2 -g -D_POSIX_C_SOURCE=200809L
LDFLAGS  = -lm

BUILD_DIR  = build
SIM_DIR    = simulation
SIM_SRCS   = $(SIM_DIR)/main.c $(SIM_DIR)/detector.c $(SIM_DIR)/display.c
SIM_OBJS   = $(patsubst $(SIM_DIR)/%.c,$(BUILD_DIR)/%.o,$(SIM_SRCS))
SIM_TARGET = $(BUILD_DIR)/sim_tinydrone

TEST_DIR   = tests
TEST_SRCS  = $(wildcard $(TEST_DIR)/*.c)
TEST_BINS  = $(patsubst $(TEST_DIR)/%.c,$(BUILD_DIR)/test_%,$(TEST_SRCS))

.PHONY: all sim test clean

all: sim

sim: $(SIM_TARGET)

$(SIM_TARGET): $(SIM_OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "  BUILD  $@"
	@echo "  Run: ./$(SIM_TARGET)"

$(BUILD_DIR)/%.o: $(SIM_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(SIM_DIR) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

test: $(TEST_BINS)
	@passed=0; total=0; \
	for t in $(TEST_BINS); do \
		./$$t && passed=$$((passed+1)); \
		total=$$((total+1)); \
	done; \
	echo "$$passed/$$total tests passed"

$(BUILD_DIR)/test_%: $(TEST_DIR)/%.c $(SIM_DIR)/detector.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(SIM_DIR) $< $(SIM_DIR)/detector.c -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)
