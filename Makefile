.PHONY: eval clean help eval-ultralow eval-low eval-medium eval-high setup-swap activate-swap resize-swap activate

# Default variables
CHECKPOINT ?= ./checkpoints/Molmo-7B-D-0924
DEVICE_BATCH_SIZE ?= 1
SEQ_LEN ?= 512
MAX_CROPS ?= 4
NPROC ?= 1
MAX_EXAMPLES ?= -1
SWAP_SIZE ?= 32G

# Ultra-low memory evaluation (minimal RAM usage)
eval-ultralow:
	@echo "Running ultra-low memory evaluation on $(TASK)..."
	@echo "Settings: batch=1, seq_len=1, max_crops=1, fp16"
	torchrun --nproc-per-node $(NPROC) launch_scripts/eval_downstream.py \
		$(CHECKPOINT) $(TASK) \
		--save_to_checkpoint_dir \
		--fsdp \
		--device_batch_size=1 \
		--seq_len=1 \
		--max_crops=1 \
		--max_examples=$(MAX_EXAMPLES) \
		fsdp.precision=pure \
		precision=amp_fp16

# Low memory evaluation (very conservative)
eval-low:
	@echo "Running low memory evaluation on $(TASK)..."
	@echo "Settings: batch=1, seq_len=512, max_crops=4, fp16"
	torchrun --nproc-per-node $(NPROC) launch_scripts/eval_downstream.py \
		$(CHECKPOINT) $(TASK) \
		--save_to_checkpoint_dir \
		--fsdp \
		--device_batch_size=1 \
		--seq_len=512 \
		--max_crops=4 \
		--max_examples=$(MAX_EXAMPLES) \
		fsdp.precision=pure \
		precision=amp_fp16

# Medium memory evaluation (balanced)
eval-medium:
	@echo "Running medium memory evaluation on $(TASK)..."
	@echo "Settings: batch=1, seq_len=1024, max_crops=8, fp16"
	torchrun --nproc-per-node $(NPROC) launch_scripts/eval_downstream.py \
		$(CHECKPOINT) $(TASK) \
		--save_to_checkpoint_dir \
		--fsdp \
		--device_batch_size=1 \
		--seq_len=1024 \
		--max_crops=8 \
		--max_examples=$(MAX_EXAMPLES) \
		fsdp.precision=pure \
		precision=amp_fp16

# High memory evaluation (requires 16GB+ RAM)
eval-high:
	@echo "Running high memory evaluation on $(TASK)..."
	@echo "Settings: batch=2, seq_len=1536, max_crops=12, fp16"
	torchrun --nproc-per-node $(NPROC) launch_scripts/eval_downstream.py \
		$(CHECKPOINT) $(TASK) \
		--save_to_checkpoint_dir \
		--fsdp \
		--device_batch_size=2 \
		--seq_len=1536 \
		--max_crops=12 \
		--max_examples=$(MAX_EXAMPLES) \
		fsdp.precision=pure \
		precision=amp_fp16

# High-resolution evaluation (original --high_res flag, requires 32GB+ RAM)
eval-highres:
	@echo "Running high-resolution evaluation on $(TASK)..."
	@echo "WARNING: This requires significant RAM (32GB+)"
	torchrun --nproc-per-node $(NPROC) launch_scripts/eval_downstream.py \
		$(CHECKPOINT) $(TASK) \
		--save_to_checkpoint_dir \
		--high_res \
		--fsdp \
		--device_batch_size=$(DEVICE_BATCH_SIZE) \
		--max_examples=$(MAX_EXAMPLES) \
		fsdp.precision=pure \
		precision=amp_fp16

# Custom evaluation with all parameters configurable
eval:
	@echo "Running custom evaluation on $(TASK)..."
	@echo "Settings: batch=$(DEVICE_BATCH_SIZE), seq_len=$(SEQ_LEN), max_crops=$(MAX_CROPS)"
	torchrun --nproc-per-node $(NPROC) launch_scripts/eval_downstream.py \
		$(CHECKPOINT) $(TASK) \
		--save_to_checkpoint_dir \
		--fsdp \
		--device_batch_size=$(DEVICE_BATCH_SIZE) \
		--seq_len=$(SEQ_LEN) \
		--max_crops=$(MAX_CROPS) \
		--max_examples=$(MAX_EXAMPLES) \
		fsdp.precision=pure \
		precision=amp_fp16

# Convenience shortcuts for specific tasks
chart_qa:
	@$(MAKE) eval-low TASK=chart_qa

chart_qa-ultra:
	@$(MAKE) eval-ultralow TASK=chart_qa

chart_qa-medium:
	@$(MAKE) eval-medium TASK=chart_qa

# Test with limited examples (for debugging)
test-chart_qa:
	@$(MAKE) eval-ultralow TASK=chart_qa MAX_EXAMPLES=10

# Setup swap file (one-time setup)
setup-swap:
	@echo "Setting up $(SWAP_SIZE) swap file..."
	@if [ -f /swapfile ]; then \
		echo "Swap file already exists at /swapfile"; \
		ls -lh /swapfile; \
	else \
		sudo fallocate -l $(SWAP_SIZE) /swapfile && \
		sudo chmod 600 /swapfile && \
		sudo mkswap /swapfile && \
		sudo swapon /swapfile && \
		echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab && \
		echo "Swap file created and activated!"; \
	fi
	@echo ""
	@swapon --show
	@echo ""
	@free -h

# Resize swap file
resize-swap:
	@echo "Resizing swap to $(SWAP_SIZE)..."
	@echo "Current swap status:"
	@swapon --show
	@echo ""
	@echo "Turning off swap..."
	@sudo swapoff /swapfile || true
	@echo "Removing old swap file..."
	@sudo rm -f /swapfile
	@echo "Creating new $(SWAP_SIZE) swap file..."
	@sudo fallocate -l $(SWAP_SIZE) /swapfile
	@sudo chmod 600 /swapfile
	@sudo mkswap /swapfile
	@sudo swapon /swapfile
	@echo ""
	@echo "New swap status:"
	@swapon --show
	@echo ""
	@free -h

# Activate swap (run after WSL restart)
activate-swap:
	@echo "Activating swap file..."
	@sudo swapon /swapfile 2>/dev/null && echo "Swap activated!" || echo "Swap already active or failed to activate"
	@echo ""
	@swapon --show
	@echo ""
	@free -h

# Monitor system resources during evaluation
monitor:
	@echo "Monitoring system resources (Ctrl+C to stop)..."
	@watch -n 1 'echo "=== Memory Usage ==="; free -h; echo ""; echo "=== GPU Usage ==="; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits'

# Clean cache and temporary files
clean:
	@echo "Cleaning Python cache and temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete!"

# Help command
help:
	@echo "Molmo Evaluation Makefile - Memory-Optimized Evaluation Commands"
	@echo ""
	@echo "=== Quick Start ==="
	@echo "  make chart_qa              - Evaluate chart_qa with low memory settings"
	@echo "  make chart_qa-ultra        - Evaluate chart_qa with ultra-low memory (for 8GB RAM)"
	@echo "  make test-chart_qa         - Quick test with 10 examples"
	@echo ""
	@echo "=== Memory Profiles ==="
	@echo "  make eval-ultralow TASK=<task>  - Ultra-low memory (8GB RAM)"
	@echo "                                     batch=1, seq_len=256, crops=1"
	@echo ""
	@echo "  make eval-low TASK=<task>       - Low memory (8-12GB RAM) - RECOMMENDED"
	@echo "                                     batch=1, seq_len=512, crops=4"
	@echo ""
	@echo "  make eval-medium TASK=<task>    - Medium memory (12-16GB RAM)"
	@echo "                                     batch=1, seq_len=1024, crops=8"
	@echo ""
	@echo "  make eval-high TASK=<task>      - High memory (16-24GB RAM)"
	@echo "                                     batch=2, seq_len=1536, crops=12"
	@echo ""
	@echo "  make eval-highres TASK=<task>   - Original high-res (32GB+ RAM)"
	@echo "                                     batch=2, seq_len=4096, crops=36"
	@echo ""
	@echo "=== Custom Evaluation ==="
	@echo "  make eval TASK=<task> SEQ_LEN=<len> MAX_CROPS=<n> DEVICE_BATCH_SIZE=<n>"
	@echo ""
	@echo "=== Available Variables ==="
	@echo "  CHECKPOINT           - Model checkpoint path (default: ./checkpoints/Molmo-7B-D-0924)"
	@echo "  TASK                 - Task to evaluate (e.g., chart_qa, doc_qa, text_vqa)"
	@echo "  DEVICE_BATCH_SIZE    - Batch size per device (default: 1)"
	@echo "  SEQ_LEN              - Maximum sequence length (default: 512)"
	@echo "  MAX_CROPS            - Maximum image crops (default: 4)"
	@echo "  MAX_EXAMPLES         - Limit examples for testing (default: -1, all)"
	@echo "  NPROC                - Number of processes (default: 1)"
	@echo ""
	@echo "=== Examples ==="
	@echo "  make chart_qa                           # Low memory mode"
	@echo "  make chart_qa-ultra                     # Ultra-low memory mode"
	@echo "  make eval-medium TASK=doc_qa            # Medium memory for doc_qa"
	@echo "  make eval TASK=text_vqa SEQ_LEN=768     # Custom settings"
	@echo "  make test-chart_qa                      # Quick test with 10 examples"
	@echo ""
	@echo "=== Utilities ==="
	@echo "  make activate          - Show commands to activate venv and set MOLMO_DATA_DIR"
	@echo "  make setup-swap        - Create swap file (default: 32GB, set SWAP_SIZE to change)"
	@echo "  make resize-swap       - Resize existing swap (default: 32GB, set SWAP_SIZE to change)"
	@echo "  make activate-swap     - Activate swap after WSL restart"
	@echo "  make monitor           - Monitor system resources in real-time"
	@echo "  make clean             - Remove Python cache files"
	@echo "  make help              - Show this help message"
	@echo ""
	@echo "=== Swap Management Examples ==="
	@echo "  make setup-swap SWAP_SIZE=32G           # Create 32GB swap"
	@echo "  make resize-swap SWAP_SIZE=64G          # Resize to 64GB"
	@echo "  make activate-swap                      # Reactivate after WSL restart"
	@echo ""
	@echo "=== Memory Optimization Tips ==="
	@echo "  1. Start with 'eval-ultralow' or 'eval-low' if you have 8GB RAM"
	@echo "  2. Use MAX_EXAMPLES=10 for quick testing before full runs"
	@echo "  3. Monitor RAM with 'make monitor' in another terminal"
	@echo "  4. If OOM occurs, reduce SEQ_LEN and MAX_CROPS further"
	@echo "  5. All modes use fp16 precision to save 50% memory vs fp32"
	@echo "  6. Run 'make activate-swap' after WSL restart to enable swap"
