
clean:
	find . | grep -E "(__pycache__|\.pyc|\.lprof)" | xargs rm -rf

install:
	pip install -r requirements.txt


lab:
	jupyter lab --ip=0.0.0.0 --port=5000 --no-browser

help:
	cat Makefile

