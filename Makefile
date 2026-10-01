# Creates the virtual environment and installs required packages
setup:
	python3 -m venv .venv
	./.venv/bin/pip install -r requirements.txt

# Runs the analysis script
run:
	python analysis.py

# Runs the Polars version of the analysis script (includes the pandas vs
# polars benchmark)
run-polars:
	python analysis_polars.py

# Deletes cached files and generated charts for a fresh run
clean:
	rm -rf __pycache__
	rm -f graphs/*.png

# Formats all Python files with black
format:
	black .

# Checks code style with flake8 and formatting with black (changes nothing)
lint:
	flake8 .
	black --check .

# Runs the test suite
test:
	pytest -v
