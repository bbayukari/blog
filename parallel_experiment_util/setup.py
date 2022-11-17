# this is a setup file for python package 'parallel'
from setuptools import setup, find_packages

setup(
    name='ParallelExperiment',
    author='Ang',
    packages=find_packages(),
    description="ParallelExperiment: use for parallel experiment",
    install_requires=[
        "numpy",
        "pandas"
    ]
)
