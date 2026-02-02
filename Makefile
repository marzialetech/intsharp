# IntSharp Makefile
.PHONY: pdf clean run install

export PATH := /Library/TeX/texbin:$(PATH)

# -------------------------------------------------------------------------
# Simulation
# -------------------------------------------------------------------------

# Run simulation with default or specified config
run:
	python run.py config.yaml

# Install Python dependencies
install:
	pip install -r requirements.txt

# -------------------------------------------------------------------------
# Manuscript
# -------------------------------------------------------------------------

# Build main manuscript PDF
pdf:
	cd manuscript && latexmk -pdf -bibtex -outdir=build main.tex && open build/main.pdf

# Clean build artifacts (keeps figs and source)
clean:
	rm -rf manuscript/build results
