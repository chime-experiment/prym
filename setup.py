from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

# load the requirements from requirements.txt while removing the environment marks
with open(path.join(here, "requirements.txt")) as f:
    requirements = [line.split(";")[0].rstrip() for line in f]

setup(
    name="prym",
    version="0.1",
    description="Convert results of PromQL queries into numpy arrays or pandas dataframes",
    url="https://github.com/chime-experiment/prym",
    author="Richard Shaw",
    python_requires=">=3.6",
    py_modules=["prym"],
    install_requires=requirements,
    packages=find_packages(),
)
