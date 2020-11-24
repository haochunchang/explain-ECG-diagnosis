#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='ECG_diagnosis',
    version='0.1.0',
    description='Training models for ECG signal diagnosis',
    author='Hao Chun Chang',
    author_email='changhaochun84@gmail.com',
    url='https://github.com/haochunchang/explain-ECG-diagnosis',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)
