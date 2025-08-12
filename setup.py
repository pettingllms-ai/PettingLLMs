from setuptools import setup, find_packages
import os
import sys

setup(
    name='pettingllms',
    version='0.1',
    package_dir={'': '.'},
    packages=find_packages(include=['pettingllms']),
    author='pettingllms Team',
    author_email='',
    acknowledgements='',
    description='',
    install_requires=[], 
    package_data={'pettingllms': ['*/*.md']},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
    ]
)