# IntSharp Makefile
.PHONY: clean run install test sweep plot propose lean-build

# -------------------------------------------------------------------------
# Simulation
# -------------------------------------------------------------------------

# Run simulation with default or specified config
run:
	python run.py config.yaml

# Install Python dependencies
install:
	pip install -r requirements.txt

# Run unit tests
test:
	pytest tests/ -v

# -------------------------------------------------------------------------
# Sharpening analysis
# -------------------------------------------------------------------------

# Mass parameter sweep (eps_target x strength) for all methods
sweep:
	python scripts/sharpening_sweep.py --workers 4 --n-eps 25 --n-str 25

# Generate stability region heatmaps from sweep data
plot:
	python scripts/plot_stability_regions.py

# Propose new sharpening terms via LLM (requires OPENAI_API_KEY)
propose:
	python scripts/propose_sharpening.py --n-proposals 1

# -------------------------------------------------------------------------
# Lean formalization
# -------------------------------------------------------------------------

# Build Lean project (fetch Mathlib + typecheck)
lean-build:
	cd lean && lake build

# Clean build artifacts
clean:
	rm -rf results
