default: lint test

test:
	scripts/test

lint:
	scripts/lint

.PHONY: default test lint
