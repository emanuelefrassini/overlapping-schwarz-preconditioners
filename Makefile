.PHONY: test demo benchmark all

test:
	python -m pytest -q

demo:
	MPLBACKEND=Agg python scripts/run_demo.py

benchmark:
	MPLBACKEND=Agg python scripts/run_benchmarks.py

all: test demo benchmark
