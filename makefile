test:
	make unit-test
	make acceptance-test

unit-test:
	poetry run pytest -v --cov=transformer_lens/ --cov-report=term-missing --cov-branch tests/unit

acceptance-test:
	poetry run pytest -v --cov=transformer_lens/ --cov-report=term-missing --cov-branch tests/acceptance
