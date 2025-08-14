# Compiler
CXX := g++

# Common flags
CXXFLAGS := -Wall -Wextra -std=c++17 -I. -MMD -MP -fopenmp

# Directories
SRC_DIR := examples
BUILD_DIR := build

# Source files
SRC := $(wildcard $(SRC_DIR)/*.cpp)

# Build mode: release by default
MODE ?= release

# Per-mode settings
ifeq ($(MODE),release)
    BUILD_SUBDIR := $(BUILD_DIR)
    CXXFLAGS += -O2
else ifeq ($(MODE),debug)
    BUILD_SUBDIR := $(BUILD_DIR)/debug
    CXXFLAGS += -g -O0
endif

# Executables: build/mode/foo.cpp -> build/mode/foo
EXE := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_SUBDIR)/%,$(SRC))

# Dependency files
DEP := $(EXE:%=%.d)

# Default target: build all executables
all: $(EXE)

# Rule: build an exe from a single .cpp
$(BUILD_SUBDIR)/%: $(SRC_DIR)/%.cpp | $(BUILD_SUBDIR)
	$(CXX) $(CXXFLAGS) $< -o $@

# Create build directory if missing
$(BUILD_SUBDIR):
	mkdir -p $(BUILD_SUBDIR)

# Run all executables with arguments "500 128"
run: all
	@for exe in $(EXE); do \
		name=$$(basename $$exe); \
		echo ""; \
		echo "---- $$name ----"; \
		"$$exe" 500 128; \
	done

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)/*

# Phony targets
.PHONY: all clean debug release run

# Explicit targets for debug/release
release:
	$(MAKE) MODE=release all

debug:
	$(MAKE) MODE=debug all

# Include dependency files if they exist
-include $(DEP)

