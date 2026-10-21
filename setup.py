# setup.py
# Developed by Hungmin Kim 
# github.com/ 
# please cite our paper if you feel this code useful : arxiv.org/abs/ ~ 
from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='trinity',
    version='1.0.0',
    packages=find_packages(include=['trinity', 'trinity.*']),
    install_requires=requirements,
)