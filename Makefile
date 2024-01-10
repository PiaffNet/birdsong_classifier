reinstall_package:
	@pip uninstall -y birdsong || :
	@pip install -e .

#################### PACKAGE ACTIONS ###################
run_slicing:
	python -c 'from birdsong.interface.main import slicing; slicing()'

run_preprocess:
	python -c 'from birdsong.interface.main import preprocess; preprocess()'

run_train:
	python -c 'from birdsong.interface.main import train; train()'

run_predict:
	python -c 'from birdsong.interface.main import predict; predict()'

run_preprocess_and_train:
	python -c 'from birdsong.interface.main import preprocess_and_train; preprocess_and_train()'


run_api:
	uvicorn birdsong.api.fast:app --reload

#################### TESTS ACTIONS ###################
test_config_file:
	@pytest tests/test_config.py

test_slicer_class:
	@pytest tests/test_slicer.py

test_preprocess_class:
	@pytest tests/test_to_image.py
