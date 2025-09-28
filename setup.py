# setup.py
from setuptools import setup
from Cython.Build import cythonize
import os

setup(
    ext_modules = cythonize(os.path.join("Tokenizer", "CharToken_fast.pyx"))
)