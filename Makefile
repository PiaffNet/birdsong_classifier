install:
	@pip install -e .

#################### PACKAGE ACTIONS ###################
run_preprocess:
	python -c 'from code.interface.main import preprocess; preprocess()'
