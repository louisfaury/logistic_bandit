from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='logbexp',
    version='1.0',
    packages=find_packages(exclude=["docs", "tests"]),
    python_requires=">=3.8",
    install_requires=['numpy', 'matplotlib', 'joblib', 'scipy', 'pandas'],
    url='https://github.com/louisfaury/logistic_bandit',
    license='Apache License 2.0',
    author='Louis Faury',
    author_email='l.faury@hotmail.fr',
    description='Logistic Bandit Experiments',
    long_description=long_description,
    long_description_content_type="text/markdown",
)
