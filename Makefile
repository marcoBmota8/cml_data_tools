# Project Makefile

.PHONY: clean-pyc uninstall install tests
clean-pyc:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.py[co]" -exec rm {} +

uninstall:
	pip uninstall -y cml-data-tools
	rm -rf src/*.egg-info

install:
	pip install -e .

tests:
	@echo 'Run an individual test file by invoking it as a module, e.g.'
	@echo '    python -m tests.test_online_norm -v'
	@echo
	@python -m unittest discover tests -v
