from setuptools import setup, find_packages

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='birdsong',
      version="0.1",
      description="Bird species classifier by song",
      install_requires=requirements,
      packages=find_packages(),
      zip_safe=False)
