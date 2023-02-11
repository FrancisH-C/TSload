# !/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import os

from setuptools import find_packages, setup

VERSION = "0.1.0"

try:
    app_dir = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(app_dir, "README.md"), encoding="utf-8") as f:
        LONG_DESCRIPTION = "\n" + f.read()
except FileNotFoundError:
    LONG_DESCRIPTION = ""

install_requires = [
    "pandas",
    "pyarrow",
    "sphinx",
    "nbsphinx",
    "sphinx-automodapi",
    "furo",
]
extras_require = {}

setup(
    name="TSload",
    version=VERSION,
    description="",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Francis Huot-Chantal",
    author_email="",
    url="https://github.com/FrancisH-C/TSload.git",
    python_requires=">=3.8.0",
    extras_require=extras_require,
    packages=find_packages(),
    install_requires=install_requires,
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
)
