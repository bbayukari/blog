from setuptools import setup, find_packages

setup(
    name='VariableSelect',
    author='Ang',
    packages=find_packages(),
    description="VariableSelect: several algorithms of general variable selection for comparison with abess",
    install_requires=[
        "numpy",
        "nlopt",
        "cvxpy"
    ]
)
