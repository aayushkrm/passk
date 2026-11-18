UV ?= uv

.PHONY: setup lint fmt type test demo report check-results site release

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

demo:
	@echo "TODO: 'passk demo' is not implemented yet (work package P-F)"

report:
	@echo "TODO: report rendering is not implemented yet (work package P-E)"

check-results:
	@echo "TODO: check-results is not implemented yet (renders README blocks and compares them)"

site:
	@echo "TODO: site build is not implemented yet (work package P-E); output will go to build/site"

release:
	@echo "TODO: release packaging is not implemented yet"
