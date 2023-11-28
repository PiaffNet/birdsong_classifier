reinstall_package:
	@pip uninstall -y birdsong || :
	@pip install -e .

#################### PACKAGE ACTIONS ###################
run_preprocess:
	python -c 'from birdsong.interface.main import preprocess; preprocess()'

run_train:
	python -c 'from birdsong.interface.main import train; train()'

run_predict:
	python -c 'from birdsong.interface.main import predict; predict()'

run_evaluate:
	python -c 'from birdsong.interface.main import evaluate; evaluate()'

#################### TESTS ACTIONS ###################
