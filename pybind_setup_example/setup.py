from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

setup(
    name="python_example",
    version="0.0.1",
    author="",
    author_email="",
    url="",
    description="",
    long_description="",
    ext_modules=[
        Pybind11Extension("python_example",
            ["src/main.cpp"],
            # Example: passing in the version to the compiled code
            cxx_std=17,
            ),
    ],
    python_requires=">=3.7",
)
