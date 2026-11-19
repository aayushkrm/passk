UV ?= uv

.PHONY: setup lint fmt type test demo results report check-results site release dogfood

setup:
	$(UV) sync --python 3.11 --extra dev

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

fmt:
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

type:
	$(UV) run mypy src

test:
	$(UV) run pytest -m "not model"

# Scripted agents only. Output is labelled SYNTHETIC. Runs in well under 30 s.
demo:
	$(UV) run passk demo

# Derived files from committed results: results/headline.json (no hand-typed numbers).
results:
	$(UV) run python scripts/render_readme.py --headline

# Renders the README results block from results/ between the results:start/end markers.
report: results
	$(UV) run python scripts/render_readme.py

# Fails when the README block or results/headline.json differ from what results/ implies.
check-results:
	$(UV) run python scripts/render_readme.py --check

# Static Pages site from committed reports. Writes an empty-state page when none are committed.
site:
	$(UV) run passk site --results results --out build/site

# Wheel and sdist into dist/. The wheel must contain the task YAML files and site templates.
release:
	$(UV) build

# CI dogfooding: validate on a fixed smoke set, and gate on committed fixture reports.
dogfood:
	$(UV) run passk validate --smoke
	$(UV) run passk gate tests/fixtures/report_baseline.json tests/fixtures/report_minor_drop.json --policy configs/gate.yaml
	! $(UV) run passk gate tests/fixtures/report_baseline.json tests/fixtures/report_regression.json --policy configs/gate.yaml
