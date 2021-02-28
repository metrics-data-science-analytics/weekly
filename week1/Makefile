NY: check clean install run prepare notebook train

export PYTHONPATH          = $(PWD):$(PWD)/src


format: clean
	echo "Running isort"
	isort src
	echo "Running black"
	black src


check:
	echo 'Running black'
	black --check src
	echo 'Running flake'
	flake8 src

test:
	echo 'Running pytest'
	pytest -vv tests --cov src --cov-report=term-missing --cov-report=html

ci: check test
	echo 'Finish Test'

clean:
	find . | grep -E "(__pycache__|\.pyc|\.lprof)" | xargs rm -rf

install:
	pip install -r requirements.txt

prepare:
	pip install jupyter nbconvert

lab:
	jupyter lab --ip=0.0.0.0 --port=5000 --no-browser

help:
	cat Makefile

