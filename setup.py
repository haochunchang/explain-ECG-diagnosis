#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='ECG_diagnosis',
    version='0.1.0',
    description='Training models for ECG signal diagnosis',
    author='Hao Chun Chang',
    author_email='changhaochun84@gmail.com',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url='',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)
