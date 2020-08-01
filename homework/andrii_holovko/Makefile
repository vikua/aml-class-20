install:
	pip install -U pip
	pip install -r requirements.txt

setup:
	test ! -d .ve && python -m venv .ve; \
	. .ve/bin/activate; \
	make install

run:
	test -d .ve && python -m venv .ve; \
	python run_pipeline.py
