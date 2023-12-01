"""
Configuration module
"""

import os
import yaml

class Configuration:
    """
    This configuration class is extremely flexible due to a two-step init process. 
    We only instanciate a single instance of it (at the bottom if this file) so that 
    all modules can import this singleton at load time. The second initialization (which 
    happens in main.py) allows the user to input custom parameters of the config class 
    at execution time.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


with open(os.path.join(os.path.dirname(__file__), '_config.yml'), 'r') as f:
    _config = yaml.load(f, Loader=yaml.FullLoader)
    config = Configuration(**_config)
