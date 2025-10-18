# Copyright (C) 2019 Richard Hughes <richard@hughsie.com>
# SPDX-License-Identifier: GPL-3.0+

VENV=./env
PYTHON=$(VENV)/bin/python
PYTEST=$(VENV)/bin/pytest
FLASK=$(VENV)/bin/flask

setup: requirements.txt
	virtualenv --python=python3 ./env
	$(VENV)/bin/pip install -r requirements.txt

clean:
	rm -rf ./build
	rm -rf ./htmlcov

run:
	FLASK_DEBUG=1 \
	ODRS_REVIEWS_SECRET=1 \
	ODRS_CONFIG=example.cfg \
	FLASK_APP=odrs/__init__.py \
	HOME=$(CURDIR) \
	$(VENV)/bin/flask run

dbup:
	ODRS_CONFIG=example.cfg \
	FLASK_APP=odrs/__init__.py \
	$(FLASK) db upgrade

dbdown:
	ODRS_CONFIG=example.cfg \
	FLASK_APP=odrs/__init__.py \
	$(FLASK) db downgrade

dbmigrate:
	ODRS_CONFIG=example.cfg \
	FLASK_APP=odrs/__init__.py \
	$(FLASK) db migrate

check: $(PYTEST)
	$(PYTEST) \
		--cov=odrs \
		--cov-report=html
	$(PYTHON) ./pylint_test.py

blacken:
	find . -path ./env -prune -o -name '*.py' -exec ./env/bin/black --quiet {} \;

codespell:
	codespell --write-changes --builtin en-GB_to_en-US --skip \
	.git,\
	.mypy_cache,\
	.coverage,\
	*.pyc,\
	*.png,\
	*.jpg,\
	*.js,\
	*.doctree,\
	*.pdf,\
	*.gz,\
	*.ico,\
	*.csv,\
	env
