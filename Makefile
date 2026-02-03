# IntSharp Makefile
.PHONY: clean run install test

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

# Clean build artifacts
clean:
	rm -rf results
