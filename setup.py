#!/usr/bin/env python3
# -- coding: utf-8 --

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mine",
    version="0.1",
    description="Mutual information neural estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BrunoBreggia/CodigoMine.git",
    author="Bruno M. Breggia",
    author_email="bruno.breggia@uner.edu.ar",
    license="MIT",
    packages=find_packages(exclude="test"),
    keywords="Mutual-Information Neural-Networks Transfer-Entropy",
    python_requires=">=3.6",
    install_requires=[
        "torch",
        "tqdm",
        "numpy",
        "matplotlib",
    ],
    test_suite="nose.collector",
    tests_require=["nose", "nose-cover3"],
    zip_safe=False,
)
