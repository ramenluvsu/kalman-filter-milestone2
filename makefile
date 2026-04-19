# =============================================================================
#  Makefile — Kalman Filter Milestone 3 (FINAL)
# =============================================================================

# ── toolchain ────────────────────────────────────────────────────────────────
CC      = riscv64-linux-gnu-gcc
QEMU    = qemu-riscv64

# ── flags ─────────────────────────────────────────────────────────────────────
CFLAGS  = -O0 -g -static -march=rv64gc -mabi=lp64d
LDFLAGS = -static -lm

# ── directories ───────────────────────────────────────────────────────────────
SRC_DIR    = src
DATA_DIR   = data
OUTPUT_DIR = output

# ── phony targets ─────────────────────────────────────────────────────────────
.PHONY: all clean run run_lkf run_ekf verify dirs help

# ── default build ─────────────────────────────────────────────────────────────
all: dirs lkf ekf
	@echo ""
	@echo "========================================"
	@echo "  Build complete: lkf and ekf binaries"
	@echo "========================================"

# ── create output directory ───────────────────────────────────────────────────
dirs:
	@mkdir -p $(OUTPUT_DIR)

# ── LKF ───────────────────────────────────────────────────────────────────────
lkf: $(SRC_DIR)/lkf_main.c $(SRC_DIR)/lkf_asm.s
	@echo "[LKF] Compiling..."
	$(CC) $(CFLAGS) $^ -o lkf $(LDFLAGS)

# ── EKF ───────────────────────────────────────────────────────────────────────
ekf: $(SRC_DIR)/ekf_main.c $(SRC_DIR)/ekf_asm.s
	@echo "[EKF] Compiling..."
	$(CC) $(CFLAGS) $^ -o ekf $(LDFLAGS)

# ── run both ──────────────────────────────────────────────────────────────────
run: all run_lkf run_ekf
	@echo ""
	@echo "========================================"
	@echo "  Outputs generated in $(OUTPUT_DIR)/"
	@echo "========================================"

# ── run LKF ───────────────────────────────────────────────────────────────────
run_lkf: lkf
	@echo "[LKF] Running..."
	$(QEMU) ./lkf
	@echo "[LKF] Output → $(OUTPUT_DIR)/lkf_asm_output.csv"

# ── run EKF ───────────────────────────────────────────────────────────────────
run_ekf: ekf
	@echo "[EKF] Running..."
	$(QEMU) ./ekf
	@echo "[EKF] Output → $(OUTPUT_DIR)/ekf_asm_output.csv"

# ── verification ──────────────────────────────────────────────────────────────
verify: run
	@echo "[VERIFY] Comparing outputs..."
	@python3 simulation/verify.py \
		$(OUTPUT_DIR)/lkf_asm_output.csv \
		$(OUTPUT_DIR)/lkf_cpp_output.csv \
		$(OUTPUT_DIR)/ekf_asm_output.csv \
		$(OUTPUT_DIR)/ekf_cpp_output.csv
	@echo "[VERIFY] Done."

# ── clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -f lkf ekf
	@echo "[CLEAN] Binaries removed (outputs preserved)"

# ── help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "Targets:"
	@echo "  make all       : build binaries"
	@echo "  make run       : run both filters"
	@echo "  make verify    : run + compare outputs"
	@echo "  make clean     : remove binaries"