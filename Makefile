# IntSharp Makefile
.PHONY: clean run install

# -------------------------------------------------------------------------
# Simulation
# -------------------------------------------------------------------------

# Run simulation with default or specified config
run:
	python run.py config.yaml

# Install Python dependencies
install:
	pip install -r requirements.txt

# Clean build artifacts
clean:
	rm -rf results
