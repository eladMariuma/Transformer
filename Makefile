PYTHON := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: venv install train sample clean

venv:
	python -m venv .venv

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

train: install
	$(PYTHON) src/train.py --epochs 10 --batch-size 64 --steps 100

sample: train
	cat data/sample_pairs.txt

clean:
	rm -rf .venv data/sample_pairs.txt
