from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup, find_packages

setup(
    name="statistic_model",
    packages=find_packages(),
    version="0.0.2",
    author="",
    author_email="",
    url="",
    description="",
    long_description="",
    ext_modules=[
        Pybind11Extension("statistic_model_pybind",
            ["src/main.cpp"],
            include_dirs=["/data/home/wangzz/.local/include"],
            # Example: passing in the version to the compiled code
            cxx_std=17,
            extra_compile_args=["-O3", "-Werror", "-DNDEBUG"],
            ),
    ],
    install_requires=["numpy", "jax"],
    python_requires=">=3.7",
)
