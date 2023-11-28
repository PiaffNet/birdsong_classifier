import os
import yaml

for key, value in yaml.load(open(os.path.join(os.path.dirname(__file__), 'config.yml'), 'r'),
                            Loader=yaml.FullLoader).items():
	globals()[key] = value
