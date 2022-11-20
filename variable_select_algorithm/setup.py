from setuptools import setup, find_packages

setup(
    name='variable_select_algorithm',
    author='Ang',
    packages=find_packages(),
    description="variable_select_algorithm: several algorithms of general variable selection for comparison with abess",
    install_requires=[
        "numpy",
        "nlopt",
        "cvxpy",
        "jax",
    ]
)
